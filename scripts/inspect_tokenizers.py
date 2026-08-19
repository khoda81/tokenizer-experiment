from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

import wandb
from tokenizer_experiment.data import mb, split_tokenizer_fit
from tokenizer_experiment.inspection import (
    bpe_merge_split_rows,
    emitted_token_rows,
    summarize_bpe_splits,
    tunstall_split_rows,
)
from tokenizer_experiment.tunstall import BPETokenizer, EmpiricalTunstallTokenizer

ARTIFACTS_DIR = Path("artifacts")


def print_token_rows(name: str, rows: list[dict], top_n: int) -> None:
    print(f"\n{name}: top {top_n} emitted tokens")
    print("  probability   count  bytes  token")
    for row in rows[:top_n]:
        print(
            f"  {row['probability']:10.4%}  {row['count']:7,d}  "
            f"{row['bytes']:5d}  {row['piece']}"
        )


def print_tunstall_splits(rows: list[dict]) -> None:
    print("\nTunstall expanded prefixes (all internal nodes except root)")
    print(
        "  mass/token  visits  seen/256   H(next)   2^H   H/255   prefix -> top continuations"
    )
    for row in rows:
        continuation = ", ".join(
            f"{item['piece']}:{item['conditional_probability']:.1%}"
            for item in row["top_continuations"][:4]
        )
        print(
            f"  {row['mass_per_emitted_token']:9.4f}  {row['visits']:7,d}  "
            f"{row['observed_children']:3d}/256  "
            f"{row['next_byte_entropy_bits']:8.3f}  "
            f"{row['effective_branching']:5.1f}  "
            f"{row['entropy_per_added_vocab_slot']:7.4f}  "
            f"{row['prefix']} -> {continuation}"
        )


def print_bpe_splits(rows: list[dict], top_n: int) -> None:
    by_support = sorted(rows, key=lambda row: row["merged_occurrences"], reverse=True)
    print(f"\nBPE merge tests: {top_n} highest-support learned merges")
    print("  merged_occ  left_occ      q    h2(q)  merge")
    for row in by_support[:top_n]:
        print(
            f"  {row['merged_occurrences']:10,d}  {row['left_occurrences']:8,d}  "
            f"{row['q_followed_by_right']:6.1%}  {row['binary_entropy_bits']:7.3f}  "
            f"{row['left']} + {row['right']} -> {row['merged']}"
        )

    balanced = sorted(
        rows,
        key=lambda row: (row["binary_entropy_bits"], row["merged_occurrences"]),
        reverse=True,
    )
    print(f"\nBPE merge tests: {top_n} most balanced among supported merges")
    print("  merged_occ  left_occ      q    h2(q)  merge")
    for row in balanced[:top_n]:
        print(
            f"  {row['merged_occurrences']:10,d}  {row['left_occurrences']:8,d}  "
            f"{row['q_followed_by_right']:6.1%}  {row['binary_entropy_bits']:7.3f}  "
            f"{row['left']} + {row['right']} -> {row['merged']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect what BPE and Tunstall actually learned on the tokenizer-fit corpus."
    )
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--tokenizer-fit-mb", type=float, default=2.0)
    parser.add_argument(
        "--tunstall-mode", choices=["boundary", "empirical", "iid"], default="boundary"
    )
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--min-bpe-support", type=int, default=100)
    parser.add_argument(
        "--output", default=str(ARTIFACTS_DIR / "tokenizer-inspection.json")
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
    fit_text = fit_raw.decode("utf-8")

    actual_vocab = EmpiricalTunstallTokenizer.legal_vocab_size(args.vocab_size)
    print(f"fit corpus: {mb(len(fit_raw)):.2f} MB; shared model vocab={actual_vocab}")

    print(f"Training {args.tunstall_mode} Tunstall tokenizer ...")
    tunstall = EmpiricalTunstallTokenizer.train(
        fit_raw,
        requested_vocab_size=actual_vocab,
        mode=args.tunstall_mode,
    )
    print("Training BPE tokenizer ...")
    bpe = BPETokenizer.train(fit_text, vocab_size=actual_vocab)

    inspect_end = tunstall.align_utf8_boundaries(fit_raw, [1.0])[0]
    inspect_raw = fit_raw[:inspect_end]
    print(
        f"inspection stream: {mb(len(inspect_raw)):.2f} MB "
        f"({len(fit_raw) - len(inspect_raw)} trailing bytes dropped for phrase alignment)"
    )

    tun_tokens = emitted_token_rows(tunstall, inspect_raw, top_n=args.top)
    bpe_tokens = emitted_token_rows(bpe, inspect_raw, top_n=args.top)
    tun_splits = tunstall_split_rows(tunstall, inspect_raw)
    bpe_splits = bpe_merge_split_rows(
        bpe,
        inspect_raw,
        min_left_occurrences=args.min_bpe_support,
    )
    bpe_summary = summarize_bpe_splits(bpe_splits)

    print_token_rows("Tunstall", tun_tokens, args.top)
    print_token_rows("BPE", bpe_tokens, args.top)
    print_tunstall_splits(tun_splits)

    print("\nBPE binary-split summary")
    for key, value in bpe_summary.items():
        print(f"  {key}: {value:.4f}")
    print_bpe_splits(bpe_splits, args.top)

    payload = {
        "dataset_config": args.dataset_config,
        "fit_bytes": len(inspect_raw),
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": actual_vocab,
        "tunstall_mode": args.tunstall_mode,
        "tunstall_tokens": tun_tokens,
        "bpe_tokens": bpe_tokens,
        "tunstall_splits": tun_splits,
        "bpe_split_summary": bpe_summary,
        "bpe_splits": bpe_splits,
        "notes": {
            "bpe_q": (
                "For learned merge A+B, q is estimated on raw corpus occurrences as "
                "count(A+B)/count(A). It tests the binary-continuation intuition; it is "
                "not a reconstruction of historical BPE trainer pair counts."
            ),
            "tunstall_entropy_per_slot": (
                "H(next byte | expanded prefix)/255: local branch entropy divided by the "
                "255 extra leaves consumed by one full byte-tree expansion."
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
            "tunstall_mode": args.tunstall_mode,
            "min_bpe_support": args.min_bpe_support,
        },
        tags=["tokenizer-inspection", "bpe", "tunstall"],
    ) as run:
        run.log({f"bpe_split/{key}": value for key, value in bpe_summary.items()})
        artifact = wandb.Artifact(
            name="bpe-tunstall-inspection",
            type="tokenizer-inspection",
            metadata={
                "tunstall_mode": args.tunstall_mode,
                "vocab_size": actual_vocab,
                "fit_bytes": len(inspect_raw),
            },
        )
        artifact.add_file(str(output_path), name="inspection.json")
        run.log_artifact(artifact)

    print(
        f"\nWrote {output_path} and logged W&B artifact bpe-tunstall-inspection"
    )


if __name__ == "__main__":
    main()
