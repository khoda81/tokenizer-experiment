from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import wandb
from tokenizer_experiment import ExperimentConfig, run_experiment

ARTIFACTS_DIR = Path("artifacts")


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
        description="Compare tokenizer families by true online prequential codelength."
    )
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument(
        "--tunstall-mode", choices=["boundary", "empirical", "iid"], default="boundary"
    )
    p.add_argument(
        "--bunstall-modes",
        type=parse_bunstall_modes,
        default=parse_bunstall_modes("frequency,entropy"),
        help="Comma-separated Bunstall modes to include; empty string disables Bunstall.",
    )
    p.add_argument("--tokenizer-fit-mb", type=float, default=2.0)
    p.add_argument("--max-preq-mb", type=float, default=0.0)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="AdamW learning rate; online default is deliberately aggressive.",
    )
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log telemetry every N datums; does not affect training or coding.",
    )
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
        "context": args.context,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "log_every": args.log_every,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    config = ExperimentConfig(**kwargs)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    curve_rows: list[list[object]] = []
    columns = [
        "model",
        "datum",
        "raw_bytes",
        "cumulative_tokens",
        "cumulative_bits",
        "bits_per_byte",
        "datum_bytes",
        "datum_tokens",
        "datum_bits_per_byte",
        "optimizer_steps",
        "elapsed_seconds",
    ]

    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        # W&B itself creates a `wandb/` child under this directory.
        dir=str(ARTIFACTS_DIR),
        config=asdict(config),
        save_code=True,
        tags=["online-prequential", "tokenization", config.tunstall_mode, "bunstall"],
    ) as run:

        def on_progress(model_name: str, point: dict) -> None:
            key = model_name.replace("-", "_")
            row = [
                model_name,
                point["datum"],
                point["cumulative_bytes"],
                point["cumulative_tokens"],
                point["cumulative_bits"],
                point["cumulative_bits_per_byte"],
                point["datum_bytes"],
                point["datum_tokens"],
                point["datum_bits_per_byte"],
                point["optimizer_steps"],
                point["elapsed_seconds"],
            ]
            curve_rows.append(row)
            run.log(
                {
                    f"{key}/datum": point["datum"],
                    f"{key}/raw_bytes": point["cumulative_bytes"],
                    f"{key}/cumulative_tokens": point["cumulative_tokens"],
                    f"{key}/cumulative_bits": point["cumulative_bits"],
                    f"{key}/cumulative_bits_per_byte": point[
                        "cumulative_bits_per_byte"
                    ],
                    f"{key}/datum_bits_per_byte": point["datum_bits_per_byte"],
                    f"{key}/optimizer_steps": point["optimizer_steps"],
                }
            )

        payload = run_experiment(config, on_progress=on_progress)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        results_artifact = wandb.Artifact(
            name="prequential-results",
            type="experiment-results",
            description="True online prequential tokenizer comparison results.",
            metadata={
                "protocol": "online-prequential",
                "dataset_config": config.dataset_config,
                "vocab_size": payload["metadata"]["actual_vocab_size"],
                "seed": config.seed,
                "lr": config.lr,
            },
        )
        results_artifact.add_file(str(output_path), name="results.json")
        run.log_artifact(results_artifact)

        table = wandb.Table(columns=columns, data=curve_rows)
        run.log(
            {
                "code_curve/table": table,
                "code_curve/by_raw_bytes": wandb.plot.line(
                    table,
                    x="raw_bytes",
                    y="bits_per_byte",
                    stroke="model",
                    title="Online prequential bits / raw byte",
                ),
                "code_curve/by_optimizer_steps": wandb.plot.line(
                    table,
                    x="optimizer_steps",
                    y="bits_per_byte",
                    stroke="model",
                    title="Online prequential code vs optimizer steps",
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
    print(f"\nWrote {output_path} and logged W&B artifact prequential-results")


if __name__ == "__main__":
    main()
