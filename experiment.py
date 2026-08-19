from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from prequential import ModelConfig, TrainConfig, run_block_prequential
from tunstall import BPETokenizer, EmpiricalTunstallTokenizer


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


def tokenizer_stats(tokenizer, text: str) -> dict:
    ids = tokenizer.encode(text)
    counts = np.bincount(
        np.asarray(ids, dtype=np.int64), minlength=tokenizer.vocab_size
    )
    probs = counts[counts > 0] / counts.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    max_entropy = math.log2(tokenizer.vocab_size)
    raw_bytes = len(text.encode("utf-8"))
    return {
        "tokens": len(ids),
        "bytes": raw_bytes,
        "bytes_per_token": raw_bytes / len(ids),
        "unigram_entropy_bits_per_token": entropy,
        "entropy_fraction_of_uniform": entropy / max_entropy,
        "distinct_tokens_used": int((counts > 0).sum()),
    }


def parse_fractions(value: str) -> list[float]:
    vals = [float(x) for x in value.split(",") if x.strip()]
    if not vals or vals[-1] != 1.0:
        raise argparse.ArgumentTypeError("fractions must end in 1.0")
    if vals[0] <= 0 or any(a >= b for a, b in itertools.pairwise(vals)):
        raise argparse.ArgumentTypeError("fractions must be increasing in (0, 1]")
    return vals


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare BPE vs Tunstall-style prefix-free tokens by WikiText block-prequential codelength."
    )
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Requested model vocab; snapped to 257 + 255k (byte-tree leaves + separate EOS).",
    )
    p.add_argument(
        "--tunstall-mode", choices=["boundary", "empirical", "iid"], default="boundary"
    )
    p.add_argument("--tokenizer-fit-mb", type=float, default=2.0)
    p.add_argument(
        "--max-preq-mb",
        type=float,
        default=0.0,
        help="Maximum prequential-stream size in MB; 0 (default) uses all remaining train bytes.",
    )
    p.add_argument(
        "--fractions",
        type=parse_fractions,
        default=parse_fractions("0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.0"),
        help="Block endpoints. Defaults to logarithmic checkpoints for the stored code curve.",
    )
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="results.json")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    actual_vocab = EmpiricalTunstallTokenizer.legal_vocab_size(args.vocab_size)
    print(
        f"Requested vocab {args.vocab_size}; using exact shared model vocab {actual_vocab} "
        f"(= 257 + 255k: prefix-free byte leaves + one separate EOS token)"
    )
    print(f"Loading Salesforce/wikitext / {args.dataset_config} ...")
    ds = load_dataset("Salesforce/wikitext", args.dataset_config, split="train")
    full_text = "\n".join(ds["text"])
    raw = full_text.encode("utf-8")
    print(f"train split: {mb(len(raw)):.2f} MB UTF-8")

    fit_n = int(args.tokenizer_fit_mb * 1_000_000)
    fit_raw, preq_raw = split_tokenizer_fit(raw, fit_n)
    if args.max_preq_mb > 0:
        preq_raw = cap_utf8(preq_raw, int(args.max_preq_mb * 1_000_000))
    fit_text = fit_raw.decode("utf-8")
    print(
        f"tokenizer fit: {mb(len(fit_raw)):.2f} MB; candidate prequential stream: {mb(len(preq_raw)):.2f} MB"
    )

    print(f"\nTraining {args.tunstall_mode} Tunstall-style tokenizer ...")
    t0 = time.perf_counter()
    tunstall = EmpiricalTunstallTokenizer.train(
        fit_raw,
        requested_vocab_size=actual_vocab,
        mode=args.tunstall_mode,
    )
    tunstall.assert_prefix_free()
    print(
        f"Tunstall tokenizer built in {time.perf_counter() - t0:.1f}s; "
        f"phrase leaves={tunstall.phrase_vocab_size}; max phrase={tunstall.max_phrase_bytes()} bytes"
    )

    # A byte-only prefix-free phrase tree cannot flush an arbitrary partial
    # phrase at a finite message boundary. Move every requested prequential cut
    # slightly backward to a completed Tunstall phrase that is also a UTF-8
    # boundary. BPE uses these exact same raw byte ranges.
    boundaries = tunstall.align_utf8_boundaries(preq_raw, args.fractions)
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

    print("\nTokenizer diagnostics on the aligned prequential stream:")
    tun_stats = tokenizer_stats(tunstall, preq_text)
    bpe_stats = tokenizer_stats(bpe, preq_text)
    for name, stats in [("tunstall", tun_stats), ("bpe", bpe_stats)]:
        print(
            f"  {name:8s}: {stats['bytes_per_token']:.3f} bytes/token, "
            f"H(T)/log2(V)={stats['entropy_fraction_of_uniform']:.4f}, "
            f"used={stats['distinct_tokens_used']}/{actual_vocab}"
        )

    model_cfg = ModelConfig(
        context=args.context,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    results = []
    for name, tokenizer in [("bpe", bpe), (f"tunstall-{args.tunstall_mode}", tunstall)]:
        print(f"\n{'=' * 72}\nPREQUENTIAL: {name}\n{'=' * 72}")
        result = run_block_prequential(
            name=name,
            tokenizer=tokenizer,
            raw=preq_raw,
            boundaries=boundaries,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            device=device,
        )
        results.append(result)

    metadata = {
        "dataset": "Salesforce/wikitext",
        "dataset_config": args.dataset_config,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": actual_vocab,
        "tunstall_phrase_vocab_size": tunstall.phrase_vocab_size,
        "eos_is_separate_token": True,
        "tunstall_mode": args.tunstall_mode,
        "tokenizer_fit_bytes": len(fit_raw),
        "prequential_bytes": len(preq_raw),
        "requested_fractions": args.fractions,
        "aligned_boundaries": boundaries,
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
        "training_passes_per_prefix": 1,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "tokenizer_stats": {
            "bpe": bpe_stats,
            f"tunstall-{args.tunstall_mode}": tun_stats,
        },
    }
    payload = {"metadata": metadata, "results": results}
    Path(args.output).write_text(json.dumps(payload, indent=2))

    print("\nFINAL")
    for r in results:
        print(f"  {r['name']:20s} {r['prequential_bits_per_byte']:.6f} bits/byte")
    delta = (
        results[1]["prequential_bits_per_byte"]
        - results[0]["prequential_bits_per_byte"]
    )
    print(f"  Tunstall - BPE       {delta:+.6f} bits/byte")
    print(f"\nWrote {args.output} (including per-model code_curve traces)")


if __name__ == "__main__":
    main()
