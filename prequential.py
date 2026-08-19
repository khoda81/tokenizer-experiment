from __future__ import annotations

import itertools
import math
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import TinyGPT


class TokenWindows(Dataset):
    def __init__(self, ids: list[int], context: int):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context = context
        self.n = max(0, math.ceil((len(ids) - 1) / context))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        start = i * self.context
        chunk = self.ids[start : start + self.context + 1]
        x = chunk[:-1]
        y = chunk[1:]
        valid = len(y)
        if valid < self.context:
            x_pad = torch.zeros(self.context, dtype=torch.long)
            y_pad = torch.full((self.context,), -100, dtype=torch.long)
            x_pad[:valid] = x
            y_pad[:valid] = y
            return x_pad, y_pad
        return x, y


@dataclass
class ModelConfig:
    context: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    max_train_steps: int = 0
    grad_clip: float = 1.0
    seed: int = 1337


@dataclass
class StageResult:
    train_fraction: float
    eval_fraction_end: float
    train_bytes: int
    eval_bytes: int
    train_tokens: int
    eval_tokens: int
    bits: float
    bits_per_byte: float
    train_seconds: float
    eval_seconds: float
    optimizer_steps: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_model(
    vocab_size: int, cfg: ModelConfig, device: torch.device, seed: int
) -> TinyGPT:
    seed_everything(seed)
    model = TinyGPT(
        vocab_size=vocab_size,
        context=cfg.context,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    )
    return model.to(device)


def train_prefix(
    model: TinyGPT,
    ids: list[int],
    cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
) -> tuple[int, float]:
    ds = TokenWindows(ids, cfg.context)
    if len(ds) == 0:
        return 0, 0.0
    g = torch.Generator()
    g.manual_seed(train_cfg.seed)
    loader = DataLoader(
        ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        generator=g,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    model.train()
    steps = 0
    start_time = time.perf_counter()
    stop = False
    for epoch in range(train_cfg.epochs):
        bar = tqdm(
            loader, desc=f"train epoch {epoch + 1}/{train_cfg.epochs}", leave=False
        )
        for x, y in bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, model.vocab_size), y.reshape(-1)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            steps += 1
            bar.set_postfix(loss=f"{loss.item():.3f}")
            if train_cfg.max_train_steps and steps >= train_cfg.max_train_steps:
                stop = True
                break
        if stop:
            break
    return steps, time.perf_counter() - start_time


@torch.inference_mode()
def score_block_bits(
    model: TinyGPT,
    ids: list[int],
    cfg: ModelConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    if not ids:
        return 0.0, 0.0
    # The block is treated as a fresh message. Its first token has no previous
    # token context, so send it under the uniform prior. Remaining tokens are
    # scored causally by the Transformer.
    total_bits = math.log2(model.vocab_size)
    ds = TokenWindows(ids, cfg.context)
    if len(ds) == 0:
        return total_bits, 0.0
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    nats = 0.0
    start_time = time.perf_counter()
    for x, y in tqdm(loader, desc="score", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context(device):
            logits = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                y.reshape(-1),
                reduction="sum",
            )
        nats += float(loss_sum)
    total_bits += nats / math.log(2.0)
    return total_bits, time.perf_counter() - start_time


def safe_utf8_boundaries(raw: bytes, fractions: list[float]) -> list[int]:
    out: list[int] = []
    prev = 0
    for frac in fractions:
        n = len(raw) if frac >= 1.0 else round(len(raw) * frac)
        while n > prev:
            try:
                raw[prev:n].decode("utf-8")
                break
            except UnicodeDecodeError as exc:
                # We only expect the cut to split the final UTF-8 codepoint.
                if exc.end == len(raw[prev:n]):
                    n -= 1
                else:
                    raise
        if n <= prev and frac < 1.0:
            raise ValueError("prequential boundary collapsed; use a larger corpus")
        out.append(n)
        prev = n
    return out


def run_block_prequential(
    *,
    name: str,
    tokenizer,
    raw: bytes,
    fractions: list[float],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
) -> dict:
    if not fractions or fractions[-1] != 1.0:
        raise ValueError("fractions must end in 1.0")
    if any(a >= b for a, b in itertools.pairwise(fractions, fractions[1:])):
        raise ValueError("fractions must be strictly increasing")

    boundaries = safe_utf8_boundaries(raw, fractions)
    actual_fractions = [b / len(raw) for b in boundaries]
    vocab = tokenizer.vocab_size

    # First block is transmitted with a uniform code, as in block-prequential
    # coding before a learner has any observations.
    first = raw[: boundaries[0]].decode("utf-8")
    first_ids = tokenizer.encode(first)
    initial_bits = len(first_ids) * math.log2(vocab)
    initial = StageResult(
        train_fraction=0.0,
        eval_fraction_end=actual_fractions[0],
        train_bytes=0,
        eval_bytes=boundaries[0],
        train_tokens=0,
        eval_tokens=len(first_ids),
        bits=initial_bits,
        bits_per_byte=initial_bits / boundaries[0],
        train_seconds=0.0,
        eval_seconds=0.0,
        optimizer_steps=0,
    )
    stages = [initial]
    print(f"[{name}] uniform block: {initial.bits_per_byte:.4f} bits/byte")

    for i in range(len(boundaries) - 1):
        train_end = boundaries[i]
        eval_end = boundaries[i + 1]
        train_text = raw[:train_end].decode("utf-8")
        eval_text = raw[train_end:eval_end].decode("utf-8")
        train_ids = tokenizer.encode(train_text)
        eval_ids = tokenizer.encode(eval_text)

        model = make_model(vocab, model_cfg, device, train_cfg.seed)
        print(
            f"[{name}] stage {i + 1}: train={train_end / 1e6:.2f} MB "
            f"({len(train_ids):,} tok), eval={(eval_end - train_end) / 1e6:.2f} MB "
            f"({len(eval_ids):,} tok)"
        )
        steps, train_seconds = train_prefix(
            model, train_ids, model_cfg, train_cfg, device
        )
        bits, eval_seconds = score_block_bits(
            model, eval_ids, model_cfg, train_cfg.batch_size, device
        )
        stage = StageResult(
            train_fraction=actual_fractions[i],
            eval_fraction_end=actual_fractions[i + 1],
            train_bytes=train_end,
            eval_bytes=eval_end - train_end,
            train_tokens=len(train_ids),
            eval_tokens=len(eval_ids),
            bits=bits,
            bits_per_byte=bits / (eval_end - train_end),
            train_seconds=train_seconds,
            eval_seconds=eval_seconds,
            optimizer_steps=steps,
        )
        stages.append(stage)
        print(f"[{name}] stage {i + 1}: {stage.bits_per_byte:.4f} bits/byte")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_bits = sum(s.bits for s in stages)
    total_bytes = sum(s.eval_bytes for s in stages)
    return {
        "name": name,
        "vocab_size": vocab,
        "total_bits": total_bits,
        "total_bytes": total_bytes,
        "prequential_bits_per_byte": total_bits / total_bytes,
        "stages": [asdict(s) for s in stages],
    }
