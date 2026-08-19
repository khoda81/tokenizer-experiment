from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import wandb
from datasets import load_dataset

from tokenizer_experiment.experiment import mb, split_tokenizer_fit, tokenizer_stats
from tokenizer_experiment.inspection import display_bytes
from tokenizer_experiment.sparse_prefix import SparsePrefixTokenizer
from tokenizer_experiment.tunstall import EmpiricalTunstallTokenizer

ARTIFACTS_DIR = Path("artifacts")


def token_rows(tokenizer: SparsePrefixTokenizer, raw: bytes, top_n: int) -> list[dict]:
    ids = tokenizer.encode_bytes(raw)
    counts = Counter(ids)
    total = len(ids)
    rows = []
    for token_id, count in counts.most_common(top_n):
        piece = tokenizer.token_piece(token_id)
        rows.append(
            {
                "token_id": token_id,
                "count": count,
                "probability": count / total,
                "bytes": len(piece),
                "piece": display_bytes(piece),
                "hex": piece.hex(),
            }
        )
    return rows


def expansion_rows(tokenizer: SparsePrefixTokenizer) -> list[dict]:
    return [
        {
            "rank": rank,
            "parent": display_bytes(expansion.parent),
            "child": display_bytes(expansion.child),
            "byte": expansion.byte,
            "child_occurrences": expansion.child_occurrences,
            "residual_occurrences": expansion.residual_occurrences,
            "q": expansion.q,
            "binary_entropy_bits": expansion.binary_entropy_bits,
            "score": expansion.score,
        }
        for rank, expansion in enumerate(tokenizer.expansions)
    ]


def summarize_expansions(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    q = np.asarray([row["q"] for row in rows], dtype=np.float64)
    h = np.asarray([row["binary_entropy_bits"] for row in rows], dtype=np.float64)
    support = np.asarray([row["child_occurrences"] for row in rows], dtype=np.float64)
    return {
        "count": float(len(rows)),
        "mean_q": float(q.mean()),
        "median_q": float(np.median(q)),
        "mean_binary_entropy_bits": float(h.mean()),
        "median_binary_entropy_bits": float(np.median(h)),
        "support_weighted_binary_entropy_bits": float(np.average(h, weights=support)),
        "fraction_q_between_0.25_and_0.75": float(np.mean((q >= 0.25) & (q <= 0.75))),
    }


def print_tokens(rows: list[dict], top_n: int) -> None:
    print(f"\nBunstall: top {top_n} emitted tokens")
    print("  probability   count  bytes  token")
    for row in rows[:top_n]:
        print(
            f"  {row['probability']:10.4%}  {row['count']:7,d}  "
            f"{row['bytes']:5d}  {row['piece']}"
        )


def print_expansions(title: str, rows: list[dict], top_n: int) -> None:
    print(f"\n{title}")
    print("  rank   child_occ  residual      q    h2(q)  parent -> child")
    for row in rows[:top_n]:
        print(
            f"  {row['rank']:4d}  {row['child_occurrences']:10,d}  "
            f"{row['residual_occurrences']:8,d}  {row['q']:6.1%}  "
            f"{row['binary_entropy_bits']:7.3f}  "
            f"{row['parent']} -> {row['child']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the sparse-prefix ('Bunstall') tokenizer prototype."
    )
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--tokenizer-fit-mb", type=float, default=2.0)
    parser.add_argument("--mode", choices=["entropy", "frequency"], default="entropy")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--output", default=str(ARTIFACTS_DIR / "bunstall-inspection.json")
    )
    parser.add_argument("--wandb-project", default="tokenizer-experiment")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    args = parser.parse_args()

    print(f"Loading Salesforce/wikitext / {args.dataset_config} ...")
    ds = load_dataset("Salesforce/wikitext", args.dataset_config, split="train")
    raw = "\n".join(ds["text"]).encode("utf-8")
    fit_raw, _ = split_tokenizer_fit(raw, int(args.tokenizer_fit_mb * 1_000_000))

    actual_vocab = EmpiricalTunstallTokenizer.legal_vocab_size(args.vocab_size)
    print(
        f"fit corpus: {mb(len(fit_raw)):.2f} MB; "
        f"Bunstall mode={args.mode}; shared model vocab={actual_vocab}"
    )

    start = time.perf_counter()
    tokenizer = SparsePrefixTokenizer.train(
        fit_raw,
        vocab_size=actual_vocab,
        mode=args.mode,
    )
    elapsed = time.perf_counter() - start
    print(
        f"Bunstall built in {elapsed:.1f}s; "
        f"max phrase={tokenizer.max_phrase_bytes()} bytes"
    )

    stats = tokenizer_stats(tokenizer, fit_raw.decode("utf-8"))
    print("\nTokenizer diagnostics on fit corpus")
    print(f"  bytes/token:       {stats['bytes_per_token']:.3f}")
    print(f"  H(T)/log2(V):      {stats['entropy_fraction_of_uniform']:.4f}")
    print(f"  unigram bits/byte: {stats['unigram_bits_per_byte']:.4f}")
    print(f"  vocab used:        {stats['distinct_tokens_used']}/{actual_vocab}")

    tokens = token_rows(tokenizer, fit_raw, args.top)
    expansions = expansion_rows(tokenizer)
    summary = summarize_expansions(expansions)
    print_tokens(tokens, args.top)

    print("\nBunstall binary-split summary")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")

    by_support = sorted(
        expansions, key=lambda row: row["child_occurrences"], reverse=True
    )
    print_expansions("Highest-support promoted continuations", by_support, args.top)

    by_balance = sorted(
        expansions,
        key=lambda row: (row["binary_entropy_bits"], row["child_occurrences"]),
        reverse=True,
    )
    print_expansions("Most balanced promoted continuations", by_balance, args.top)

    payload = {
        "dataset_config": args.dataset_config,
        "fit_bytes": len(fit_raw),
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": actual_vocab,
        "mode": args.mode,
        "build_seconds": elapsed,
        "tokenizer_stats": stats,
        "top_tokens": tokens,
        "expansion_summary": summary,
        "expansions": expansions,
        "notes": {
            "semantics": (
                "SparsePrefixTokenizer uses greedy longest-match parsing. Parents remain "
                "valid tokens when one-byte children are promoted, so its vocabulary is "
                "not prefix-free. This isolates sparse prefix refinement from Tunstall's "
                "mandatory 256-way expansion."
            ),
            "training_counts": (
                "Continuation counts are arbitrary-position corpus counts, not exact "
                "emitted-boundary counts. This is a cheap structural prototype."
            ),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    wandb_dir = ARTIFACTS_DIR / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        dir=str(wandb_dir),
        job_type="tokenizer-inspection",
        config={
            "dataset_config": args.dataset_config,
            "vocab_size": actual_vocab,
            "tokenizer_fit_mb": args.tokenizer_fit_mb,
            "bunstall_mode": args.mode,
        },
        tags=["tokenizer-inspection", "bunstall", args.mode],
    ) as run:
        run.log({f"tokenizer/{key}": value for key, value in stats.items() if isinstance(value, (int, float))})
        artifact = wandb.Artifact(
            name=f"bunstall-{args.mode}-inspection",
            type="tokenizer-inspection",
            metadata={
                "mode": args.mode,
                "vocab_size": actual_vocab,
                "fit_bytes": len(fit_raw),
            },
        )
        artifact.add_file(str(output_path), name="inspection.json")
        run.log_artifact(artifact)

    print(
        f"\nWrote {output_path} and logged W&B artifact "
        f"bunstall-{args.mode}-inspection"
    )


if __name__ == "__main__":
    main()
