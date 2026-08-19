from __future__ import annotations

import argparse
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
    if vals[0] <= 0 or any(a >= b for a, b in zip(vals, vals[1:])):
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
        help="Requested size; snapped to 257 + 256k.",
    )
    p.add_argument(
        "--tunstall-mode", choices=["boundary", "empirical", "iid"], default="boundary"
    )
    p.add_argument("--tokenizer-fit-mb", type=float, default=2.0)
    p.add_argument(
        "--max-preq-mb", type=float, default=4.0, help="0 = all remaining train bytes"
    )
    p.add_argument(
        "--fractions",
        type=parse_fractions,
        default=parse_fractions("0.05,0.15,0.35,0.65,1.0"),
    )
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="0 = full epochs; useful for smoke tests",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="results.json")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    actual_vocab = EmpiricalTunstallTokenizer.legal_vocab_size(args.vocab_size)
    print(
        f"Requested vocab {args.vocab_size}; using exact shared vocab {actual_vocab} (= 257 + 256k)"
    )
    print(f"Loading Salesforce/wikitext / {args.dataset_config} ...")
    ds = load_dataset("Salesforce/wikitext", args.dataset_config, split="train")
    # Preserve blank rows and explicit row boundaries. Both tokenizers see the
    # exact same UTF-8 stream.
    full_text = "\n".join(ds["text"])
    raw = full_text.encode("utf-8")
    print(f"train split: {mb(len(raw)):.2f} MB UTF-8")

    fit_n = int(args.tokenizer_fit_mb * 1_000_000)
    fit_raw, preq_raw = split_tokenizer_fit(raw, fit_n)
    if args.max_preq_mb > 0:
        preq_raw = cap_utf8(preq_raw, int(args.max_preq_mb * 1_000_000))
    fit_text = fit_raw.decode("utf-8")
    preq_text = preq_raw.decode("utf-8")
    print(
        f"tokenizer fit: {mb(len(fit_raw)):.2f} MB; prequential stream: {mb(len(preq_raw)):.2f} MB"
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
        f"Tunstall tokenizer built in {time.perf_counter() - t0:.1f}s; max phrase={tunstall.max_phrase_bytes()} bytes"
    )

    print("\nTraining byte-level BPE tokenizer ...")
    t0 = time.perf_counter()
    bpe = BPETokenizer.train(fit_text, vocab_size=actual_vocab)
    print(f"BPE tokenizer built in {time.perf_counter() - t0:.1f}s")

    print("\nTokenizer diagnostics on the prequential stream:")
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
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_train_steps=args.max_train_steps,
        seed=args.seed,
    )

    results = []
    for name, tokenizer in [("bpe", bpe), (f"tunstall-{args.tunstall_mode}", tunstall)]:
        print(f"\n{'=' * 72}\nPREQUENTIAL: {name}\n{'=' * 72}")
        result = run_block_prequential(
            name=name,
            tokenizer=tokenizer,
            raw=preq_raw,
            fractions=args.fractions,
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
        "tunstall_mode": args.tunstall_mode,
        "tokenizer_fit_bytes": len(fit_raw),
        "prequential_bytes": len(preq_raw),
        "fractions": args.fractions,
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
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
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
