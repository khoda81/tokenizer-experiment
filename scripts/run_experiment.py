from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import wandb
from tokenizer_experiment import ExperimentConfig, run_experiment

ARTIFACTS_DIR = Path("artifacts")


def parse_fractions(value: str) -> list[float]:
    vals = [float(x) for x in value.split(",") if x.strip()]
    if not vals or vals[-1] != 1.0:
        raise argparse.ArgumentTypeError("fractions must end in 1.0")
    if vals[0] <= 0 or any(a >= b for a, b in itertools.pairwise(vals)):
        raise argparse.ArgumentTypeError("fractions must be increasing in (0, 1]")
    return vals


def parse_bunstall_modes(value: str) -> tuple[str, ...]:
    modes = tuple(x.strip() for x in value.split(",") if x.strip())
    invalid = set(modes) - {"entropy", "frequency"}
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown Bunstall modes: {', '.join(sorted(invalid))}"
        )
    return modes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare tokenizer families by WikiText block-prequential codelength."
    )
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument(
        "--tunstall-mode", choices=["boundary", "empirical", "iid"], default="boundary"
    )
    p.add_argument(
        "--bunstall-modes",
        type=parse_bunstall_modes,
        default=parse_bunstall_modes("entropy,frequency"),
        help="Comma-separated Bunstall modes to include; empty string disables Bunstall.",
    )
    p.add_argument("--tokenizer-fit-mb", type=float, default=2.0)
    p.add_argument("--max-preq-mb", type=float, default=0.0)
    p.add_argument(
        "--fractions",
        type=parse_fractions,
        default=parse_fractions("0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.0"),
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
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=str(ARTIFACTS_DIR / "results.json"))

    p.add_argument("--wandb-project", default="tokenizer-experiment")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode. Use disabled for a completely local run.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    kwargs = {
        "dataset_config": args.dataset_config,
        "vocab_size": args.vocab_size,
        "tunstall_mode": args.tunstall_mode,
        "bunstall_modes": args.bunstall_modes,
        "tokenizer_fit_mb": args.tokenizer_fit_mb,
        "max_preq_mb": args.max_preq_mb,
        "fractions": args.fractions,
        "context": args.context,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    config = ExperimentConfig(**kwargs)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wandb_dir = ARTIFACTS_DIR / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    curve_rows: list[list[object]] = []
    columns = [
        "model",
        "stage",
        "raw_bytes",
        "cumulative_bits",
        "bits_per_byte",
        "model_train_bytes",
        "model_train_tokens",
        "model_optimizer_steps",
        "cumulative_optimizer_steps",
        "cumulative_seconds",
    ]

    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        dir=str(wandb_dir),
        config=asdict(config),
        save_code=True,
        tags=["prequential", "tokenization", config.tunstall_mode, "bunstall"],
    ) as run:

        def on_stage(model_name: str, stage: dict) -> None:
            key = model_name.replace("-", "_")
            row = [
                model_name,
                stage["stage"],
                stage["cumulative_bytes"],
                stage["cumulative_bits"],
                stage["cumulative_bits_per_byte"],
                stage["train_bytes"],
                stage["train_tokens"],
                stage["optimizer_steps"],
                stage["cumulative_optimizer_steps"],
                stage["cumulative_seconds"],
            ]
            curve_rows.append(row)
            run.log(
                {
                    f"{key}/stage": stage["stage"],
                    f"{key}/raw_bytes": stage["cumulative_bytes"],
                    f"{key}/cumulative_bits": stage["cumulative_bits"],
                    f"{key}/cumulative_bits_per_byte": stage[
                        "cumulative_bits_per_byte"
                    ],
                    f"{key}/cumulative_optimizer_steps": stage[
                        "cumulative_optimizer_steps"
                    ],
                    f"{key}/block_bits_per_byte": stage["bits_per_byte"],
                }
            )

        payload = run_experiment(config, on_stage=on_stage)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        table = wandb.Table(columns=columns, data=curve_rows)
        run.log(
            {
                "code_curve/table": table,
                "code_curve/by_raw_bytes": wandb.plot.line(
                    table,
                    x="raw_bytes",
                    y="bits_per_byte",
                    stroke="model",
                    title="Cumulative prequential bits / raw byte",
                ),
                "code_curve/by_optimizer_steps": wandb.plot.line(
                    table,
                    x="cumulative_optimizer_steps",
                    y="bits_per_byte",
                    stroke="model",
                    title="Prequential code vs cumulative optimizer steps",
                ),
            }
        )

        bpe_result = next(result for result in payload["results"] if result["name"] == "bpe")
        bpe_bpb = bpe_result["prequential_bits_per_byte"]
        for result in payload["results"]:
            name = result["name"].replace("-", "_")
            bpb = result["prequential_bits_per_byte"]
            run.summary[f"final/{name}_bits_per_byte"] = bpb
            if result["name"] != "bpe":
                run.summary[f"final/{name}_minus_bpe_bits_per_byte"] = bpb - bpe_bpb
        run.summary["metadata"] = payload["metadata"]

    print("\nFINAL")
    for result in payload["results"]:
        bpb = result["prequential_bits_per_byte"]
        suffix = "" if result["name"] == "bpe" else f"  ({bpb - bpe_bpb:+.6f} vs BPE)"
        print(f"  {result['name']:20s} {bpb:.6f} bits/byte{suffix}")
    print(f"\nWrote {output_path} (including per-model code_curve traces)")


if __name__ == "__main__":
    main()
