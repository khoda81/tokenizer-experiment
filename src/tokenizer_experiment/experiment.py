from __future__ import annotations

import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from .model import ModelConfig
from .prequential import StageCallback, TrainConfig, run_block_prequential
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
    fractions: list[float] = field(
        default_factory=lambda: [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
    )
    context: int = 256
    d_model: int = 256
    layers: int = 4
    heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.1
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def mb(n: int) -> float:
    return n / 1_000_000


def safe_prefix(raw: bytes, max_bytes: int) -> tuple[bytes, bytes]:
    if max_bytes <= 0 or max_bytes >= len(raw):
        return raw, b""
    n = max_bytes
    while n > 0:
        try:
            prefix = raw[:n].decode("utf-8")
            return prefix.encode("utf-8"), raw[n:]
        except UnicodeDecodeError as exc:
            if exc.end == n:
                n -= 1
            else:
                raise
    raise RuntimeError("could not find UTF-8 boundary")


def split_tokenizer_fit(raw: bytes, fit_bytes: int) -> tuple[bytes, bytes]:
    if fit_bytes >= len(raw):
        raise ValueError("tokenizer-fit prefix consumes the whole dataset")
    prefix, remainder = safe_prefix(raw, fit_bytes)
    if not remainder:
        raise ValueError("no bytes remain for prequential evaluation")
    return prefix, remainder


def cap_utf8(raw: bytes, max_bytes: int) -> bytes:
    if max_bytes <= 0 or len(raw) <= max_bytes:
        return raw
    prefix, _ = safe_prefix(raw, max_bytes)
    return prefix


def tokenizer_stats(tokenizer, text: str) -> dict[str, Any]:
    ids = tokenizer.encode(text)
    counts = np.bincount(
        np.asarray(ids, dtype=np.int64), minlength=tokenizer.vocab_size
    )
    probs = counts[counts > 0] / counts.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    raw_bytes = len(text.encode("utf-8"))
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
    on_stage: StageCallback | None = None,
) -> dict[str, Any]:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    actual_vocab = EmpiricalTunstallTokenizer.legal_vocab_size(config.vocab_size)
    print(
        f"Requested vocab {config.vocab_size}; using exact shared model vocab {actual_vocab} "
        f"(= 257 + 255k: prefix-free byte leaves + one separate EOS token)"
    )
    print(f"Loading Salesforce/wikitext / {config.dataset_config} ...")
    ds = load_dataset("Salesforce/wikitext", config.dataset_config, split="train")
    full_text = "\n".join(ds["text"])
    raw = full_text.encode("utf-8")
    print(f"train split: {mb(len(raw)):.2f} MB UTF-8")

    fit_raw, preq_raw = split_tokenizer_fit(
        raw, int(config.tokenizer_fit_mb * 1_000_000)
    )
    if config.max_preq_mb > 0:
        preq_raw = cap_utf8(preq_raw, int(config.max_preq_mb * 1_000_000))
    fit_text = fit_raw.decode("utf-8")
    print(
        f"tokenizer fit: {mb(len(fit_raw)):.2f} MB; "
        f"candidate prequential stream: {mb(len(preq_raw)):.2f} MB"
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

    # Preserve the original experiment's Tunstall-safe raw-byte checkpoints.
    # BPE and Bunstall can encode arbitrary finite blocks, so they use exactly
    # the same raw byte ranges. A later experiment can additionally require cuts
    # to be token boundaries for every tokenizer.
    boundaries = tunstall.align_utf8_boundaries(preq_raw, config.fractions)
    encoded_end = boundaries[-1]
    dropped_tail = len(preq_raw) - encoded_end
    preq_raw = preq_raw[:encoded_end]
    preq_text = preq_raw.decode("utf-8")
    print(
        f"aligned prequential stream: {mb(encoded_end):.2f} MB; "
        f"dropped trailing partial phrase={dropped_tail} bytes"
    )
    print("aligned cuts (MB): " + ", ".join(f"{b / 1e6:.3f}" for b in boundaries))

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

    print("\nTokenizer diagnostics on the aligned prequential stream:")
    stats_by_name: dict[str, dict[str, Any]] = {}
    for name, tokenizer in tokenizers:
        stats = tokenizer_stats(tokenizer, preq_text)
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
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        seed=config.seed,
    )

    results = []
    for name, tokenizer in tokenizers:
        print(f"\n{'=' * 72}\nPREQUENTIAL: {name}\n{'=' * 72}")
        results.append(
            run_block_prequential(
                name=name,
                tokenizer=tokenizer,
                raw=preq_raw,
                boundaries=boundaries,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                device=device,
                on_stage=on_stage,
            )
        )

    metadata = {
        "dataset": "Salesforce/wikitext",
        "dataset_config": config.dataset_config,
        "requested_vocab_size": config.vocab_size,
        "actual_vocab_size": actual_vocab,
        "tunstall_phrase_vocab_size": tunstall.phrase_vocab_size,
        "eos_is_separate_token": True,
        "tunstall_mode": config.tunstall_mode,
        "bunstall_modes": list(config.bunstall_modes),
        "tokenizer_fit_bytes": len(fit_raw),
        "prequential_bytes": len(preq_raw),
        "requested_fractions": config.fractions,
        "aligned_boundaries": boundaries,
        "boundary_policy": "tunstall-safe raw-byte cuts shared by all tokenizers",
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
        "training_passes_per_prefix": 1,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "tokenizer_stats": stats_by_name,
    }
    return {"metadata": metadata, "results": results}
