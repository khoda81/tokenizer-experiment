from __future__ import annotations

import bisect
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from .inspection import bytelevel_piece_bytes
from .model import ModelConfig
from .prequential import ProgressCallback, TrainConfig, run_stream_prequential
from .tunstall import BPETokenizer
from .unigram import ByteUnigramTokenizer


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_config: str = "wikitext-2-raw-v1"
    # Preserve the 4082-wide model used by the completed comparison. Without
    # Tunstall there is no longer a structural requirement for this size, but
    # keeping it fixed makes the focused BPE/Unigram follow-up directly comparable.
    vocab_size: int = 4082
    unigram_max_piece_length: int = 16
    tokenizer_fit_mb: float = 2.0
    max_preq_mb: float = 0.0
    update_bytes: int = 256
    context: int = 256
    d_model: int = 256
    layers: int = 4
    heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    learning_rates: tuple[float, ...] = (1e-3,)
    weight_decay: float = 0.1
    seed: int = 1337
    log_every: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def mb(n: int) -> float:
    return n / 1_000_000


def _dataset_rows(texts: list[str]) -> list[bytes]:
    """Return the exact `"\n".join(texts)` stream split at dataset-row boundaries."""
    rows: list[bytes] = []
    for i, text in enumerate(texts):
        suffix = b"\n" if i + 1 < len(texts) else b""
        raw = text.encode("utf-8") + suffix
        if raw:
            rows.append(raw)
    return rows


def _split_fit_rows(rows: list[bytes], target_bytes: int) -> tuple[bytes, list[bytes]]:
    if target_bytes <= 0:
        raise ValueError("tokenizer_fit_mb must be positive")
    total = 0
    split = 0
    while split < len(rows) and total + len(rows[split]) <= target_bytes:
        total += len(rows[split])
        split += 1
    if split == 0 or split == len(rows):
        raise ValueError("tokenizer-fit prefix must contain some, but not all, dataset rows")
    return b"".join(rows[:split]), rows[split:]


def _cap_rows(rows: list[bytes], max_bytes: int) -> list[bytes]:
    if max_bytes <= 0:
        return rows
    out: list[bytes] = []
    total = 0
    for raw in rows:
        if total + len(raw) > max_bytes:
            break
        out.append(raw)
        total += len(raw)
    if not out:
        raise ValueError("max_preq_mb is smaller than the first remaining row")
    return out


def _token_piece_bytes(tokenizer, token_id: int) -> bytes:
    if isinstance(tokenizer, BPETokenizer):
        token_text = tokenizer.tokenizer.id_to_token(token_id)
        if token_text is None or token_id == tokenizer.eos_id:
            raise ValueError(f"unexpected BPE token id in raw stream: {token_id}")
        return bytelevel_piece_bytes(token_text)
    return bytes(tokenizer.token_piece(token_id))


def _encode_stream(tokenizer, raw: bytes) -> tuple[list[int], list[int]]:
    """Tokenize once and return token-end offsets in raw UTF-8 bytes."""
    ids = tokenizer.encode(raw.decode("utf-8"))
    offsets = [0]
    for token_id in ids:
        piece = _token_piece_bytes(tokenizer, token_id)
        if not piece:
            raise ValueError("zero-byte token appeared in raw tokenization")
        offsets.append(offsets[-1] + len(piece))
    if offsets[-1] != len(raw):
        raise AssertionError(
            f"token pieces cover {offsets[-1]} bytes, expected {len(raw)}"
        )
    return ids, offsets


def _shared_update_boundaries(
    offsets_by_name: dict[str, list[int]], target_bytes: int
) -> list[int]:
    """Choose shared token boundaries near absolute raw-byte milestones."""
    if target_bytes <= 0:
        raise ValueError("update_bytes must be positive")
    if not offsets_by_name:
        raise ValueError("need token offsets")

    finals = {offsets[-1] for offsets in offsets_by_name.values()}
    if len(finals) != 1:
        raise ValueError(f"tokenizers cover different raw lengths: {sorted(finals)}")
    final = finals.pop()

    common = set(offsets_by_name[next(iter(offsets_by_name))])
    for offsets in offsets_by_name.values():
        common.intersection_update(offsets)
    common_sorted = sorted(common)
    if not common_sorted or common_sorted[0] != 0 or common_sorted[-1] != final:
        raise RuntimeError("stream start/end are not common token boundaries")

    boundaries: list[int] = []
    last = 0
    for target in range(target_bytes, final, target_bytes):
        i = bisect.bisect_left(common_sorted, target)
        if i == len(common_sorted):
            break
        boundary = common_sorted[i]
        if boundary > last and boundary < final:
            boundaries.append(boundary)
            last = boundary
    if not boundaries or boundaries[-1] != final:
        boundaries.append(final)
    return boundaries


def tokenizer_stats(tokenizer, ids: list[int], raw_bytes: int) -> dict[str, Any]:
    counts = np.bincount(
        np.asarray(ids, dtype=np.int64), minlength=tokenizer.vocab_size
    )
    probs = counts[counts > 0] / counts.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    bytes_per_token = raw_bytes / len(ids)
    return {
        "tokens": len(ids),
        "bytes": raw_bytes,
        "bytes_per_token": bytes_per_token,
        "unigram_entropy_bits_per_token": entropy,
        "unigram_bits_per_byte": entropy / bytes_per_token,
        "entropy_fraction_of_uniform": entropy / math.log2(tokenizer.vocab_size),
        "distinct_tokens_used": int((counts > 0).sum()),
    }


def run_experiment(
    config: ExperimentConfig,
    *,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if config.vocab_size < 257:
        raise ValueError("vocab_size must leave room for 256 byte symbols plus BOS")
    if not config.learning_rates or any(lr <= 0 for lr in config.learning_rates):
        raise ValueError("learning_rates must contain positive values")
    if len(set(config.learning_rates)) != len(config.learning_rates):
        raise ValueError("learning_rates must be unique")

    actual_vocab = config.vocab_size
    print(
        f"Shared model vocab {actual_vocab} "
        f"({actual_vocab - 1} source-token slots + one reserved BOS token)"
    )
    print(f"Loading Salesforce/wikitext / {config.dataset_config} ...")
    ds = load_dataset("Salesforce/wikitext", config.dataset_config, split="train")
    rows = _dataset_rows(ds["text"])
    total_bytes = sum(map(len, rows))
    print(f"train split: {mb(total_bytes):.2f} MB UTF-8 across {len(rows):,} nonempty raw rows")

    fit_raw, preq_rows = _split_fit_rows(
        rows, int(config.tokenizer_fit_mb * 1_000_000)
    )
    if config.max_preq_mb > 0:
        preq_rows = _cap_rows(preq_rows, int(config.max_preq_mb * 1_000_000))
    fit_text = fit_raw.decode("utf-8")
    preq_raw = b"".join(preq_rows)
    print(
        f"tokenizer fit: {mb(len(fit_raw)):.2f} MB; "
        f"continuous stream: {mb(len(preq_raw)):.2f} MB"
    )

    print("\nTraining byte-level Unigram LM tokenizer ...")
    t0 = time.perf_counter()
    unigram = ByteUnigramTokenizer.train(
        fit_text,
        vocab_size=actual_vocab,
        max_piece_length=config.unigram_max_piece_length,
    )
    print(
        f"Byte-Unigram built in {time.perf_counter() - t0:.1f}s; "
        f"source pieces={unigram.source_vocab_size}/{actual_vocab - 1}; "
        f"max phrase={unigram.max_phrase_bytes()} bytes"
    )

    print("\nTraining byte-level BPE tokenizer ...")
    t0 = time.perf_counter()
    bpe = BPETokenizer.train(fit_text, vocab_size=actual_vocab)
    print(f"BPE tokenizer built in {time.perf_counter() - t0:.1f}s")

    tokenizers: list[tuple[str, Any]] = [
        ("byte-unigram", unigram),
        ("bpe", bpe),
    ]

    print("\nTokenizing the continuous stream once per tokenizer ...")
    streams: dict[str, tuple[list[int], list[int]]] = {}
    stats_by_name: dict[str, dict[str, Any]] = {}
    offsets_by_name: dict[str, list[int]] = {}
    for name, tokenizer in tokenizers:
        ids, offsets = _encode_stream(tokenizer, preq_raw)
        streams[name] = (ids, offsets)
        offsets_by_name[name] = offsets
        stats = tokenizer_stats(tokenizer, ids, len(preq_raw))
        stats_by_name[name] = stats
        print(
            f"  {name:20s}: {stats['bytes_per_token']:.3f} bytes/token, "
            f"H(T)/log2(V)={stats['entropy_fraction_of_uniform']:.4f}, "
            f"unigram={stats['unigram_bits_per_byte']:.4f} bpb, "
            f"used={stats['distinct_tokens_used']}/{actual_vocab}"
        )

    raw_boundaries = _shared_update_boundaries(offsets_by_name, config.update_bytes)
    update_sizes = np.diff(np.asarray([0, *raw_boundaries], dtype=np.int64))
    print(
        f"shared optimizer updates: {len(raw_boundaries):,}; "
        f"target={config.update_bytes} raw bytes; "
        f"mean={update_sizes.mean():.1f}, min={update_sizes.min()}, max={update_sizes.max()} bytes"
    )

    token_boundaries_by_name: dict[str, list[int]] = {}
    for name, (_ids, offsets) in streams.items():
        index_by_offset = {offset: i for i, offset in enumerate(offsets)}
        token_boundaries_by_name[name] = [index_by_offset[b] for b in raw_boundaries]

    model_cfg = ModelConfig(
        context=config.context,
        d_model=config.d_model,
        n_layers=config.layers,
        n_heads=config.heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
    )

    results: list[dict[str, Any]] = []
    sweep = len(config.learning_rates) > 1
    for lr in config.learning_rates:
        train_cfg = TrainConfig(
            lr=lr,
            weight_decay=config.weight_decay,
            seed=config.seed,
        )
        for name, tokenizer in tokenizers:
            ids, _offsets = streams[name]
            run_name = f"{name}@lr={lr:g}" if sweep else name
            print(
                f"\n{'=' * 72}\nCONTINUOUS PREQUENTIAL: {name}  lr={lr:g}\n{'=' * 72}"
            )
            result = run_stream_prequential(
                name=run_name,
                tokenizer=tokenizer,
                ids=ids,
                raw_boundaries=raw_boundaries,
                token_boundaries=token_boundaries_by_name[name],
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                device=device,
                log_every=config.log_every,
                on_progress=on_progress,
            )
            result["tokenizer"] = name
            result["learning_rate"] = lr
            results.append(result)

    metadata = {
        "protocol": (
            "continuous-stream online prequential: tokenize once; preserve autoregressive "
            "context across update boundaries; score each raw segment before one update"
        ),
        "dataset": "Salesforce/wikitext",
        "dataset_config": config.dataset_config,
        "actual_vocab_size": actual_vocab,
        "reserved_special_token": "used as BOS only at the start of the continuous stream",
        "tokenizers": [name for name, _ in tokenizers],
        "unigram": {
            "implementation": "Hugging Face Tokenizers Unigram + ByteLevel",
            "source_vocab_size": unigram.source_vocab_size,
            "max_piece_length": config.unigram_max_piece_length,
            "normalization": "none",
            "pretokenizer_regex": False,
        },
        "learning_rates": list(config.learning_rates),
        "tokenizer_fit_bytes": len(fit_raw),
        "prequential_bytes": len(preq_raw),
        "stream_policy": (
            "exact newline-joined WikiText byte stream; dataset rows do not reset model context"
        ),
        "update_target_bytes": config.update_bytes,
        "shared_update_count": len(raw_boundaries),
        "update_bytes_mean": float(update_sizes.mean()),
        "update_bytes_min": int(update_sizes.min()),
        "update_bytes_max": int(update_sizes.max()),
        "model": asdict(model_cfg),
        "training": {
            "weight_decay": config.weight_decay,
            "seed": config.seed,
        },
        "optimizer_steps_per_run": len(raw_boundaries),
        "passes_over_stream_per_run": 1,
        "log_every_updates": config.log_every,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "tokenizer_stats": stats_by_name,
    }
    return {"metadata": metadata, "results": results}
