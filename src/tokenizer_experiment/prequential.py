from __future__ import annotations

import bisect
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
DEFAULT_TAIL_WINDOWS_BYTES = (250_000, 500_000, 1_000_000, 2_000_000, 4_000_000)


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.1
    seed: int = 1337


@dataclass
class OnlinePoint:
    update: int
    update_bytes: int
    update_tokens: int
    update_bits: float
    update_bits_per_byte: float
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


def _input_window(
    ids: list[int], *, x_start: int, x_end: int, bos_id: int
) -> list[int]:
    """Return conceptual ``[BOS, *ids][x_start:x_end]`` without copying ids."""
    if x_start == 0:
        if x_end <= 0:
            return []
        return [bos_id, *ids[: x_end - 1]]
    return ids[x_start - 1 : x_end - 1]


def _stream_update_step(
    model: CausalTransformer,
    optimizer: torch.optim.Optimizer,
    ids: list[int],
    *,
    start_token: int,
    end_token: int,
    bos_id: int,
    context: int,
    device: torch.device,
) -> float:
    """Score one new continuous-stream segment, then update once from that loss.

    ``ids`` is the tokenization of the *entire* continuous prequential stream.
    Tokens before ``start_token`` remain available as autoregressive context.
    Only targets in ``[start_token, end_token)`` contribute to the code/loss.

    The model cannot process more than ``context`` input positions at once. For
    a large update segment we therefore score it in subchunks of at most half a
    context window. This reserves at least half of each window for preceding
    stream context once enough history exists. Gradients from all subchunks are
    accumulated before the single optimizer step, so every probability charged
    to this update comes from the pre-update model state.
    """
    if not (0 <= start_token < end_token <= len(ids)):
        raise ValueError("invalid token update range")
    if context < 2:
        raise ValueError("context must be at least 2")

    update_tokens = end_token - start_token
    max_new_tokens = max(1, context // 2)

    optimizer.zero_grad(set_to_none=True)
    model.train()
    total_nats = 0.0

    for chunk_start in range(start_token, end_token, max_new_tokens):
        chunk_end = min(chunk_start + max_new_tokens, end_token)
        new_tokens = chunk_end - chunk_start

        # Conceptual sequence position q in [BOS, *ids] predicts q+1. Ending x
        # at `chunk_end` makes its final logit predict ids[chunk_end-1].
        x_end = chunk_end
        x_start = max(0, x_end - context)
        x_values = _input_window(ids, x_start=x_start, x_end=x_end, bos_id=bos_id)
        x = torch.tensor(x_values, dtype=torch.long, device=device)[None, :]
        targets = torch.tensor(
            ids[chunk_start:chunk_end], dtype=torch.long, device=device
        )[None, :]

        with autocast_context(device):
            logits = model(x)
            new_logits = logits[:, -new_tokens:, :]
            loss_sum = F.cross_entropy(
                new_logits.reshape(-1, model.vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )

        total_nats += loss_sum.detach().item()
        (loss_sum / update_tokens).backward()

    optimizer.step()
    return total_nats / math.log(2.0)


def _tail_rates(
    cumulative_bytes: list[int],
    cumulative_bits: list[float],
    windows: tuple[int, ...] = DEFAULT_TAIL_WINDOWS_BYTES,
) -> dict[str, dict[str, float | int]]:
    """Return exact code rates from the closest real update boundary to each tail size."""
    if len(cumulative_bytes) != len(cumulative_bits):
        raise ValueError("tail cumulative arrays differ in length")
    final_bytes = cumulative_bytes[-1]
    final_bits = cumulative_bits[-1]
    out: dict[str, dict[str, float | int]] = {}
    for requested in windows:
        if requested <= 0 or requested >= final_bytes:
            continue
        target = final_bytes - requested
        i = bisect.bisect_left(cumulative_bytes, target)
        # Both neighbouring entries are real optimizer boundaries. Pick the one
        # whose resulting tail length is closest to the requested raw-byte span.
        candidates = [i]
        if i > 0:
            candidates.append(i - 1)
        start_i = min(candidates, key=lambda j: abs(cumulative_bytes[j] - target))
        start_bytes = cumulative_bytes[start_i]
        span = final_bytes - start_bytes
        bits = final_bits - cumulative_bits[start_i]
        out[str(requested)] = {
            "requested_bytes": requested,
            "actual_bytes": span,
            "bits": bits,
            "bits_per_byte": bits / span,
        }
    return out


def run_stream_prequential(
    *,
    name: str,
    tokenizer,
    ids: list[int],
    raw_boundaries: list[int],
    token_boundaries: list[int],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    log_every: int = 100,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Online prequential code over one continuous token stream.

    Raw update boundaries are shared across tokenizers. They affect only when
    the optimizer is allowed to learn; they do not reset the Transformer
    context or retokenize the stream. Each update is scored before its single
    optimizer step.
    """
    if not ids:
        raise ValueError("need a nonempty token stream")
    if not raw_boundaries or not token_boundaries:
        raise ValueError("need update boundaries")
    if len(raw_boundaries) != len(token_boundaries):
        raise ValueError("raw/token boundary counts differ")
    if raw_boundaries[-1] <= 0 or token_boundaries[-1] != len(ids):
        raise ValueError("final boundaries must cover the whole stream")
    if log_every <= 0:
        raise ValueError("log_every must be positive")

    set_seed(train_cfg.seed)
    model = CausalTransformer(tokenizer.vocab_size, model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    total_bits = 0.0
    total_tokens = 0
    trace: list[dict[str, Any]] = []
    # Keep one tiny cumulative record per optimizer boundary so tail rates are
    # computed from the actual code, independent of telemetry/log_every.
    cumulative_raw = [0]
    cumulative_code = [0.0]
    start_time = time.perf_counter()
    prev_raw = 0
    prev_token = 0

    progress = tqdm(
        enumerate(zip(raw_boundaries, token_boundaries), start=1),
        total=len(raw_boundaries),
        desc=name,
    )
    for update_i, (raw_end, token_end) in progress:
        if raw_end <= prev_raw or token_end <= prev_token:
            raise ValueError("update boundaries must be strictly increasing")

        bits = _stream_update_step(
            model,
            optimizer,
            ids,
            start_token=prev_token,
            end_token=token_end,
            bos_id=tokenizer.eos_id,
            context=model_cfg.context,
            device=device,
        )

        update_bytes = raw_end - prev_raw
        update_tokens = token_end - prev_token
        total_bits += bits
        total_tokens += update_tokens
        cumulative_raw.append(raw_end)
        cumulative_code.append(total_bits)
        point = OnlinePoint(
            update=update_i,
            update_bytes=update_bytes,
            update_tokens=update_tokens,
            update_bits=bits,
            update_bits_per_byte=bits / update_bytes,
            cumulative_bytes=raw_end,
            cumulative_tokens=total_tokens,
            cumulative_bits=total_bits,
            cumulative_bits_per_byte=total_bits / raw_end,
            optimizer_steps=update_i,
            elapsed_seconds=time.perf_counter() - start_time,
        )

        if update_i == 1 or update_i % log_every == 0 or update_i == len(raw_boundaries):
            payload = asdict(point)
            trace.append(payload)
            if on_progress is not None:
                on_progress(name, payload)
            progress.set_postfix(
                bpb=f"{point.cumulative_bits_per_byte:.4f}",
                mb=f"{point.cumulative_bytes / 1e6:.2f}",
            )

        prev_raw = raw_end
        prev_token = token_end

    elapsed = time.perf_counter() - start_time
    return {
        "name": name,
        "protocol": "continuous-stream-online-prequential",
        "prequential_bits": total_bits,
        "prequential_bits_per_byte": total_bits / raw_boundaries[-1],
        "encoded_bytes": raw_boundaries[-1],
        "updates": len(raw_boundaries),
        "optimizer_steps": len(raw_boundaries),
        "tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "tail": _tail_rates(cumulative_raw, cumulative_code),
        "code_curve": trace,
    }
