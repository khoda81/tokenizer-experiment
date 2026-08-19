from __future__ import annotations

import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from .model import ModelConfig
from .prequential import ProgressCallback, TrainConfig, run_online_prequential
from .sparse_prefix import SparsePrefixTokenizer
from .tunstall import BPETokenizer, EmpiricalTunstallTokenizer


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_config: str = "wikitext-2-raw-v1"
    vocab_size: int = 4096
    tunstall_mode: str = "boundary"
    bunstall_modes: tuple[str, ...] = ("entropy", "frequency")
    tokenizer_fit_mb: float = 2.0
    max_preq_mb: float = 0.0
    context: int = 256
    d_model: int = 256
    layers: int = 4
    heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    lr: float = 1e-3
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
        raise ValueError("max_preq_mb is smaller than the first prequential datum")
    return out


def _group_tunstall_safe_rows(
    rows: list[bytes], tokenizer: EmpiricalTunstallTokenizer
) -> tuple[list[bytes], list[int], int]:
    """Maximize row-level datum count subject to complete Tunstall phrases.

    We close a datum at every dataset-row boundary where the continuously parsed
    Tunstall tree has returned to its root. Adjacent rows are merged only when a
    phrase straddles their boundary. The resulting raw datums are then shared by
    every tokenizer.
    """
    datums: list[bytes] = []
    rows_per_datum: list[int] = []
    buf = bytearray()
    buffered_rows = 0
    node_id = tokenizer.root

    for raw in rows:
        buf.extend(raw)
        buffered_rows += 1
        for sym in raw:
            children = tokenizer.nodes[node_id].children
            if children is None:
                raise AssertionError("expected Tunstall internal node")
            node_id = children[sym]
            if tokenizer.nodes[node_id].is_leaf:
                node_id = tokenizer.root

        if node_id == tokenizer.root:
            datums.append(bytes(buf))
            rows_per_datum.append(buffered_rows)
            buf.clear()
            buffered_rows = 0

    return datums, rows_per_datum, len(buf)


def tokenizer_stats(tokenizer, datums: list[bytes]) -> dict[str, Any]:
    counts = np.zeros(tokenizer.vocab_size, dtype=np.int64)
    raw_bytes = 0
    tokens = 0
    for raw in datums:
        ids = tokenizer.encode(raw.decode("utf-8"))
        if ids:
            counts += np.bincount(np.asarray(ids, dtype=np.int64), minlength=tokenizer.vocab_size)
        raw_bytes += len(raw)
        tokens += len(ids)

    probs = counts[counts > 0] / counts.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    bytes_per_token = raw_bytes / tokens
    return {
        "tokens": tokens,
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

    actual_vocab = EmpiricalTunstallTokenizer.legal_vocab_size(config.vocab_size)
    print(
        f"Requested vocab {config.vocab_size}; using exact shared model vocab {actual_vocab} "
        f"(4081 phrase slots + one separate EOS at this size)"
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
    candidate_bytes = sum(map(len, preq_rows))
    print(
        f"tokenizer fit: {mb(len(fit_raw)):.2f} MB; "
        f"candidate online stream: {mb(candidate_bytes):.2f} MB in {len(preq_rows):,} rows"
    )

    print(f"\nTraining {config.tunstall_mode} Tunstall-style tokenizer ...")
    t0 = time.perf_counter()
    tunstall = EmpiricalTunstallTokenizer.train(
        fit_raw, requested_vocab_size=actual_vocab, mode=config.tunstall_mode
    )
    tunstall.assert_prefix_free()
    print(
        f"Tunstall tokenizer built in {time.perf_counter() - t0:.1f}s; "
        f"phrase leaves={tunstall.phrase_vocab_size}; "
        f"max phrase={tunstall.max_phrase_bytes()} bytes"
    )

    datums, rows_per_datum, dropped_tail = _group_tunstall_safe_rows(preq_rows, tunstall)
    online_bytes = sum(map(len, datums))
    print(
        f"online datums: {len(datums):,} from {sum(rows_per_datum):,} rows; "
        f"{mb(online_bytes):.2f} MB; dropped trailing incomplete group={dropped_tail} bytes"
    )
    if rows_per_datum:
        print(
            f"rows/datum: mean={np.mean(rows_per_datum):.3f}, "
            f"max={max(rows_per_datum)}, single-row={np.mean(np.asarray(rows_per_datum) == 1):.1%}"
        )

    print("\nTraining byte-level BPE tokenizer ...")
    t0 = time.perf_counter()
    bpe = BPETokenizer.train(fit_text, vocab_size=actual_vocab)
    print(f"BPE tokenizer built in {time.perf_counter() - t0:.1f}s")

    bunstalls: dict[str, SparsePrefixTokenizer] = {}
    for mode in config.bunstall_modes:
        if mode not in {"entropy", "frequency"}:
            raise ValueError(f"unknown Bunstall mode: {mode}")
        print(f"\nTraining Bunstall-{mode} tokenizer ...")
        t0 = time.perf_counter()
        tokenizer = SparsePrefixTokenizer.train(fit_raw, actual_vocab, mode=mode)
        bunstalls[mode] = tokenizer
        print(
            f"Bunstall-{mode} built in {time.perf_counter() - t0:.1f}s; "
            f"max phrase={tokenizer.max_phrase_bytes()} bytes"
        )

    tokenizers: list[tuple[str, Any]] = [
        ("bpe", bpe),
        (f"tunstall-{config.tunstall_mode}", tunstall),
        *[(f"bunstall-{mode}", bunstalls[mode]) for mode in config.bunstall_modes],
    ]

    print("\nTokenizer diagnostics on the actual online datums:")
    stats_by_name: dict[str, dict[str, Any]] = {}
    for name, tokenizer in tokenizers:
        stats = tokenizer_stats(tokenizer, datums)
        stats_by_name[name] = stats
        print(
            f"  {name:20s}: {stats['bytes_per_token']:.3f} bytes/token, "
            f"H(T)/log2(V)={stats['entropy_fraction_of_uniform']:.4f}, "
            f"unigram={stats['unigram_bits_per_byte']:.4f} bpb, "
            f"used={stats['distinct_tokens_used']}/{actual_vocab}"
        )

    model_cfg = ModelConfig(
        context=config.context,
        d_model=config.d_model,
        n_layers=config.layers,
        n_heads=config.heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
    )
    train_cfg = TrainConfig(
        lr=config.lr,
        weight_decay=config.weight_decay,
        seed=config.seed,
    )

    results = []
    for name, tokenizer in tokenizers:
        print(f"\n{'=' * 72}\nONLINE PREQUENTIAL: {name}\n{'=' * 72}")
        results.append(
            run_online_prequential(
                name=name,
                tokenizer=tokenizer,
                datums=datums,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                device=device,
                log_every=config.log_every,
                on_progress=on_progress,
            )
        )

    metadata = {
        "protocol": "online-prequential: score datum, then update once from that same loss",
        "dataset": "Salesforce/wikitext",
        "dataset_config": config.dataset_config,
        "requested_vocab_size": config.vocab_size,
        "actual_vocab_size": actual_vocab,
        "tunstall_phrase_vocab_size": tunstall.phrase_vocab_size,
        "eos_is_separate_token": True,
        "tunstall_mode": config.tunstall_mode,
        "bunstall_modes": list(config.bunstall_modes),
        "tokenizer_fit_bytes": len(fit_raw),
        "candidate_prequential_rows": len(preq_rows),
        "online_datums": len(datums),
        "prequential_bytes": online_bytes,
        "dropped_tail_bytes": dropped_tail,
        "datum_policy": (
            "finest dataset-row groups whose ends are complete Tunstall phrase boundaries; "
            "same raw datums for every tokenizer"
        ),
        "rows_per_datum_mean": float(np.mean(rows_per_datum)) if rows_per_datum else 0.0,
        "rows_per_datum_max": max(rows_per_datum) if rows_per_datum else 0,
        "single_row_datum_fraction": float(np.mean(np.asarray(rows_per_datum) == 1))
        if rows_per_datum
        else 0.0,
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
        "batch_size_datums": 1,
        "passes_over_stream": 1,
        "optimizer_steps": len(datums),
        "log_every_datums": config.log_every,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "tokenizer_stats": stats_by_name,
    }
    return {"metadata": metadata, "results": results}
