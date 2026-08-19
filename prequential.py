from __future__ import annotations

import itertools
import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass(frozen=True)
class ModelConfig:
    context: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.1
    seed: int = 1337


@dataclass
class StageResult:
    stage: int
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
    cumulative_bytes: int
    cumulative_bits: float
    cumulative_bits_per_byte: float
    cumulative_optimizer_steps: int
    cumulative_seconds: float


class TokenWindows(Dataset):
    def __init__(self, ids: list[int], context: int):
        if len(ids) < 2:
            raise ValueError("need at least two tokens")
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context = context
        self.n = math.ceil((len(ids) - 1) / context)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.context
        end = min(start + self.context + 1, len(self.ids))
        chunk = self.ids[start:end]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def collate_windows(batch):
    max_len = max(x.shape[0] for x, _ in batch)
    xs = torch.zeros((len(batch), max_len), dtype=torch.long)
    ys = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        xs[i, : len(x)] = x
        ys[i, : len(y)] = y
    return xs, ys


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, cfg: ModelConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.context = cfg.context
        self.token = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.mlp_ratio,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # nn.Embedding defaults to N(0, 1). That is disastrous when its weight is
        # reused as the output projection: after LayerNorm the initial logits have
        # enormous variance and the untrained model is confidently random. Use the
        # small embedding initialization common in decoder-only Transformers before
        # tying input and output weights.
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
        self.head.weight = self.token.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        _, t = ids.shape
        if t > self.context:
            raise ValueError(f"sequence length {t} exceeds context {self.context}")
        positions = torch.arange(t, device=ids.device)
        h = self.token(ids) + self.pos(positions)[None, :, :]
        mask = torch.triu(
            torch.ones(t, t, device=ids.device, dtype=torch.bool), diagonal=1
        )
        h = self.transformer(h, mask=mask)
        return self.head(self.norm(h))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def train_model(
    ids: list[int],
    vocab_size: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
) -> tuple[CausalTransformer, int, float]:
    """Train a fresh model for exactly one pass over the observed prefix."""
    set_seed(train_cfg.seed)
    model = CausalTransformer(vocab_size, model_cfg).to(device)
    ds = TokenWindows(ids, model_cfg.context)
    loader = DataLoader(
        ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    model.train()
    steps = 0
    start_time = time.perf_counter()
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100
            )
        loss.backward()
        optimizer.step()
        steps += 1
    return model, steps, time.perf_counter() - start_time


@torch.no_grad()
def score_model(
    model: CausalTransformer,
    ids: list[int],
    context: int,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    if len(ids) < 2:
        return 0.0, 0.0
    model.eval()
    ds = TokenWindows(ids, context)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )
    # Every window predicts token 1..N from token 0..N-1. The first token of
    # each independently encoded evaluation block needs a uniform code because
    # this toy setup does not pass train-prefix context into scoring.
    total_bits = math.log2(model.vocab_size)
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
                ignore_index=-100,
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
    if any(a >= b for a, b in itertools.pairwise(fractions)):
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
        stage=0,
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
        cumulative_bytes=boundaries[0],
        cumulative_bits=initial_bits,
        cumulative_bits_per_byte=initial_bits / boundaries[0],
        cumulative_optimizer_steps=0,
        cumulative_seconds=0.0,
    )
    stages = [initial]
    cumulative_bits = initial_bits
    cumulative_steps = 0
    cumulative_seconds = 0.0
    print(f"[{name}] uniform block: {initial.bits_per_byte:.4f} bits/byte")

    for i in range(len(boundaries) - 1):
        train_end = boundaries[i]
        eval_end = boundaries[i + 1]
        train_text = raw[:train_end].decode("utf-8")
        eval_text = raw[train_end:eval_end].decode("utf-8")
        train_ids = tokenizer.encode(train_text)
        eval_ids = tokenizer.encode(eval_text)

        print(
            f"[{name}] stage {i + 1}/{len(boundaries) - 1}: "
            f"train={train_end / 1e6:.2f}MB ({len(train_ids):,} tok), "
            f"score={(eval_end - train_end) / 1e6:.2f}MB ({len(eval_ids):,} tok)"
        )
        model, steps, train_seconds = train_model(
            train_ids,
            vocab,
            model_cfg,
            train_cfg,
            device,
        )
        bits, eval_seconds = score_model(
            model,
            eval_ids,
            model_cfg.context,
            device,
            train_cfg.batch_size,
        )

        cumulative_bits += bits
        cumulative_steps += steps
        cumulative_seconds += train_seconds + eval_seconds
        stage = StageResult(
            stage=i + 1,
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
            cumulative_bytes=eval_end,
            cumulative_bits=cumulative_bits,
            cumulative_bits_per_byte=cumulative_bits / eval_end,
            cumulative_optimizer_steps=cumulative_steps,
            cumulative_seconds=cumulative_seconds,
        )
        stages.append(stage)
        print(
            f"[{name}]   block={stage.bits_per_byte:.4f} bits/byte; "
            f"cumulative={stage.cumulative_bits_per_byte:.4f}; "
            f"steps={steps:,}; train {train_seconds:.1f}s; score {eval_seconds:.1f}s"
        )

    # `code_curve` is deliberately redundant with `stages`: it is a stable,
    # compact plotting interface for comparing prequential code against data,
    # optimizer work, or elapsed training/scoring time without rerunning.
    code_curve = [
        {
            "stage": s.stage,
            "raw_bytes": s.cumulative_bytes,
            "fraction": s.eval_fraction_end,
            "cumulative_bits": s.cumulative_bits,
            "bits_per_byte": s.cumulative_bits_per_byte,
            "model_train_bytes": s.train_bytes,
            "model_train_tokens": s.train_tokens,
            "model_optimizer_steps": s.optimizer_steps,
            "cumulative_optimizer_steps": s.cumulative_optimizer_steps,
            "cumulative_seconds": s.cumulative_seconds,
        }
        for s in stages
    ]

    return {
        "name": name,
        "prequential_bits": cumulative_bits,
        "prequential_bits_per_byte": cumulative_bits / len(raw),
        "stages": [s.__dict__ for s in stages],
        "code_curve": code_curve,
    }
