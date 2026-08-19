from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .model import CausalTransformer, ModelConfig

ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.1
    seed: int = 1337


@dataclass
class OnlinePoint:
    datum: int
    datum_bytes: int
    datum_tokens: int
    datum_bits: float
    datum_bits_per_byte: float
    cumulative_bytes: int
    cumulative_tokens: int
    cumulative_bits: float
    cumulative_bits_per_byte: float
    optimizer_steps: int
    elapsed_seconds: float


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


def _online_datum_step(
    model: CausalTransformer,
    optimizer: torch.optim.Optimizer,
    ids: list[int],
    *,
    eos_id: int,
    context: int,
    device: torch.device,
) -> float:
    """Measure one datum's NLL, then update once from that exact loss.

    EOS is used as both the start-of-datum context and the termination target:

        <EOS> token_0 ... token_n <EOS>

    Long datums are evaluated in context-sized chunks. Gradients from all chunks
    are accumulated before the single optimizer step, so every probability used
    in the prequential code comes from the model state *before* this datum is
    learned.
    """
    sequence = [eos_id, *ids, eos_id]
    predictions = len(sequence) - 1
    if predictions <= 0:
        raise AssertionError("datum must have at least the EOS target")

    optimizer.zero_grad(set_to_none=True)
    model.train()
    total_nats = 0.0

    # Each target is scored exactly once. Context resets only for unusually long
    # datums that exceed the model's token context; the optimizer still steps
    # exactly once for the whole datum.
    for start in range(0, predictions, context):
        end = min(start + context, predictions)
        x = torch.tensor(sequence[start:end], dtype=torch.long, device=device)[None, :]
        y = torch.tensor(sequence[start + 1 : end + 1], dtype=torch.long, device=device)[
            None, :
        ]
        with autocast_context(device):
            logits = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                y.reshape(-1),
                reduction="sum",
            )
        total_nats += float(loss_sum.detach())
        # Normalize the update by datum size so a long line does not implicitly
        # get a larger learning rate than a short line.
        (loss_sum / predictions).backward()

    optimizer.step()
    return total_nats / math.log(2.0)


def run_online_prequential(
    *,
    name: str,
    tokenizer,
    datums: list[bytes],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    log_every: int = 100,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Online prequential code: loss(datum), then one update, exactly once.

    This is intentionally *not* block prequential coding. A single model is
    initialized once and walks through the data in order. Datum boundaries are
    fixed by the raw dataset before tokenization and are shared by all tokenizers.
    """
    if not datums:
        raise ValueError("need at least one online datum")
    if log_every <= 0:
        raise ValueError("log_every must be positive")

    set_seed(train_cfg.seed)
    model = CausalTransformer(tokenizer.vocab_size, model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    total_bits = 0.0
    total_bytes = 0
    total_tokens = 0
    trace: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    progress = tqdm(enumerate(datums, start=1), total=len(datums), desc=name)
    for datum_i, raw in progress:
        text = raw.decode("utf-8")
        ids = tokenizer.encode(text)
        bits = _online_datum_step(
            model,
            optimizer,
            ids,
            eos_id=tokenizer.eos_id,
            context=model_cfg.context,
            device=device,
        )

        datum_bytes = len(raw)
        total_bits += bits
        total_bytes += datum_bytes
        total_tokens += len(ids)
        point = OnlinePoint(
            datum=datum_i,
            datum_bytes=datum_bytes,
            datum_tokens=len(ids),
            datum_bits=bits,
            datum_bits_per_byte=bits / datum_bytes,
            cumulative_bytes=total_bytes,
            cumulative_tokens=total_tokens,
            cumulative_bits=total_bits,
            cumulative_bits_per_byte=total_bits / total_bytes,
            optimizer_steps=datum_i,
            elapsed_seconds=time.perf_counter() - start_time,
        )

        if datum_i == 1 or datum_i % log_every == 0 or datum_i == len(datums):
            payload = asdict(point)
            trace.append(payload)
            if on_progress is not None:
                on_progress(name, payload)
            progress.set_postfix(
                bpb=f"{point.cumulative_bits_per_byte:.4f}",
                mb=f"{point.cumulative_bytes / 1e6:.2f}",
            )

    elapsed = time.perf_counter() - start_time
    return {
        "name": name,
        "protocol": "online-prequential",
        "prequential_bits": total_bits,
        "prequential_bits_per_byte": total_bits / total_bytes,
        "encoded_bytes": total_bytes,
        "datums": len(datums),
        "optimizer_steps": len(datums),
        "tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "code_curve": trace,
    }
