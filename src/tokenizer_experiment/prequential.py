from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .model import CausalTransformer, ModelConfig

StageCallback = Callable[[str, dict[str, Any]], None]


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
        return chunk[:-1], chunk[1:]


def collate_windows(batch):
    max_len = max(x.shape[0] for x, _ in batch)
    xs = torch.zeros((len(batch), max_len), dtype=torch.long)
    ys = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        xs[i, : len(x)] = x
        ys[i, : len(y)] = y
    return xs, ys


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
    if not ids:
        return 0.0, 0.0

    # The first token of an independently encoded block has no model context.
    total_bits = math.log2(model.vocab_size)
    if len(ids) == 1:
        return total_bits, 0.0

    model.eval()
    ds = TokenWindows(ids, context)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )
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
    return total_bits + nats / math.log(2.0), time.perf_counter() - start_time


def _emit_stage(
    callback: StageCallback | None, model_name: str, stage: StageResult
) -> None:
    if callback is not None:
        callback(model_name, asdict(stage))


def run_block_prequential(
    *,
    name: str,
    tokenizer,
    raw: bytes,
    boundaries: list[int],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    on_stage: StageCallback | None = None,
) -> dict[str, Any]:
    if not boundaries or boundaries[-1] > len(raw):
        raise ValueError("invalid prequential boundaries")
    if boundaries[0] <= 0 or any(a >= b for a, b in itertools.pairwise(boundaries)):
        raise ValueError("prequential boundaries must be strictly increasing")

    actual_fractions = [b / boundaries[-1] for b in boundaries]
    vocab = tokenizer.vocab_size

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
    _emit_stage(on_stage, name, initial)

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
            train_ids, vocab, model_cfg, train_cfg, device
        )
        bits, eval_seconds = score_model(
            model, eval_ids, model_cfg.context, device, train_cfg.batch_size
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
        _emit_stage(on_stage, name, stage)
        print(
            f"[{name}]   block={stage.bits_per_byte:.4f} bits/byte; "
            f"cumulative={stage.cumulative_bits_per_byte:.4f}; "
            f"steps={steps:,}; train {train_seconds:.1f}s; score {eval_seconds:.1f}s"
        )

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

    encoded_bytes = boundaries[-1]
    return {
        "name": name,
        "prequential_bits": cumulative_bits,
        "prequential_bits_per_byte": cumulative_bits / encoded_bytes,
        "encoded_bytes": encoded_bytes,
        "stages": [asdict(s) for s in stages],
        "code_curve": code_curve,
    }
