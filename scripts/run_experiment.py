from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import wandb

from tokenizer_experiment import ExperimentConfig, run_experiment

ARTIFACTS_DIR = Path("artifacts")


def parse_learning_rates(value: str) -> tuple[float, ...]:
    try:
        rates = tuple(float(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("learning rates must be numbers") from exc
    if not rates or any(lr <= 0 for lr in rates):
        raise argparse.ArgumentTypeError("learning rates must be positive")
    if len(set(rates)) != len(rates):
        raise argparse.ArgumentTypeError("learning rates must be unique")
    return rates


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _metric_key(model_name: str) -> str:
    return (
        model_name.replace("-", "_")
        .replace("@", "_")
        .replace("=", "_")
        .replace(".", "p")
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare byte-level Unigram and BPE by continuous-stream prequential codelength."
    )
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument(
        "--vocab-size",
        type=int,
        default=4082,
        help="Shared Transformer vocabulary width; 4082 preserves the completed baseline.",
    )
    p.add_argument(
        "--unigram-max-piece-length",
        type=int,
        default=16,
        help="Maximum Byte-Unigram piece length in ByteLevel symbols/bytes.",
    )
    p.add_argument("--tokenizer-fit-mb", type=float, default=2.0)
    p.add_argument("--max-preq-mb", type=float, default=0.0)
    p.add_argument(
        "--update-bytes",
        type=int,
        default=256,
        help=(
            "Target raw bytes between optimizer updates. Actual updates move to the "
            "nearest following raw-byte position that is a token boundary for both tokenizers."
        ),
    )
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
        help="Single AdamW learning rate. Ignored when --lrs is supplied.",
    )
    p.add_argument(
        "--lrs",
        type=parse_learning_rates,
        default=None,
        help="Comma-separated LR sweep, e.g. 3e-4,1e-3,3e-3. Tokenizers are trained only once.",
    )
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log/checkpoint telemetry every N optimizer updates; does not affect coding.",
    )
    p.add_argument(
        "--artifact-every",
        type=int,
        default=1000,
        help="Log a versioned W&B progress artifact every N optimizer updates.",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=str(ARTIFACTS_DIR / "results.json"))
    p.add_argument(
        "--partial-output", default=str(ARTIFACTS_DIR / "results.partial.json")
    )

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
    if args.artifact_every <= 0:
        raise SystemExit("--artifact-every must be positive")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive")

    learning_rates = args.lrs if args.lrs is not None else (args.lr,)
    kwargs = {
        "dataset_config": args.dataset_config,
        "vocab_size": args.vocab_size,
        "unigram_max_piece_length": args.unigram_max_piece_length,
        "tokenizer_fit_mb": args.tokenizer_fit_mb,
        "max_preq_mb": args.max_preq_mb,
        "update_bytes": args.update_bytes,
        "context": args.context,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "learning_rates": learning_rates,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "log_every": args.log_every,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    config = ExperimentConfig(**kwargs)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    partial_path = Path(args.partial_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    curve_rows: list[list[object]] = []
    columns = [
        "model",
        "update",
        "raw_bytes",
        "cumulative_tokens",
        "cumulative_bits",
        "bits_per_byte",
        "update_bytes",
        "update_tokens",
        "update_bits_per_byte",
        "optimizer_steps",
        "elapsed_seconds",
    ]
    partial_state: dict[str, object] = {
        "status": "starting",
        "config": asdict(config),
        "models": {},
    }
    atomic_write_json(partial_path, partial_state)

    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        dir=str(ARTIFACTS_DIR),
        config={**asdict(config), "artifact_every": args.artifact_every},
        save_code=True,
        tags=["continuous-prequential", "byte-unigram", "bpe", "lr-sweep"],
    ) as run:

        def log_progress_artifact(model_name: str, update: int, status: str) -> None:
            artifact = wandb.Artifact(
                name="prequential-progress",
                type="experiment-progress",
                description="Incremental checkpoint from a continuous prequential run.",
                metadata={
                    "status": status,
                    "model": model_name,
                    "update": update,
                    "seed": config.seed,
                    "update_target_bytes": config.update_bytes,
                },
            )
            artifact.add_file(str(partial_path), name="results.partial.json")
            run.log_artifact(artifact)

        def on_progress(model_name: str, point: dict) -> None:
            key = _metric_key(model_name)
            row = [
                model_name,
                point["update"],
                point["cumulative_bytes"],
                point["cumulative_tokens"],
                point["cumulative_bits"],
                point["cumulative_bits_per_byte"],
                point["update_bytes"],
                point["update_tokens"],
                point["update_bits_per_byte"],
                point["optimizer_steps"],
                point["elapsed_seconds"],
            ]
            curve_rows.append(row)

            models = partial_state["models"]
            assert isinstance(models, dict)
            model_state = models.setdefault(model_name, {"code_curve": []})
            model_state["latest"] = point
            model_state["code_curve"].append(point)
            partial_state["status"] = "running"
            partial_state["current_model"] = model_name
            atomic_write_json(partial_path, partial_state)

            run.log(
                {
                    f"{key}/update": point["update"],
                    f"{key}/raw_bytes": point["cumulative_bytes"],
                    f"{key}/cumulative_tokens": point["cumulative_tokens"],
                    f"{key}/cumulative_bits": point["cumulative_bits"],
                    f"{key}/cumulative_bits_per_byte": point[
                        "cumulative_bits_per_byte"
                    ],
                    f"{key}/update_bits_per_byte": point["update_bits_per_byte"],
                    f"{key}/optimizer_steps": point["optimizer_steps"],
                }
            )

            if point["update"] == 1 or point["update"] % args.artifact_every == 0:
                log_progress_artifact(model_name, point["update"], "running")

        try:
            payload = run_experiment(config, on_progress=on_progress)
        except BaseException as exc:
            partial_state["status"] = "interrupted"
            partial_state["error_type"] = type(exc).__name__
            partial_state["error"] = str(exc)
            atomic_write_json(partial_path, partial_state)
            current_model = str(partial_state.get("current_model", "startup"))
            models = partial_state["models"]
            latest_update = 0
            if isinstance(models, dict) and current_model in models:
                latest = models[current_model].get("latest", {})
                latest_update = int(latest.get("update", 0))
            log_progress_artifact(current_model, latest_update, "interrupted")
            raise

        atomic_write_json(output_path, payload)
        partial_state["status"] = "complete"
        partial_state["final_output"] = str(output_path)
        atomic_write_json(partial_path, partial_state)

        results_artifact = wandb.Artifact(
            name="prequential-results",
            type="experiment-results",
            description="Focused Byte-Unigram vs BPE prequential LR sweep.",
            metadata={
                "protocol": "continuous-stream-online-prequential",
                "dataset_config": config.dataset_config,
                "vocab_size": payload["metadata"]["actual_vocab_size"],
                "seed": config.seed,
                "learning_rates": list(config.learning_rates),
                "update_target_bytes": config.update_bytes,
            },
        )
        results_artifact.add_file(str(output_path), name="results.json")
        results_artifact.add_file(str(partial_path), name="results.partial.json")
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
                    title="Continuous prequential bits / raw byte",
                ),
                "code_curve/by_optimizer_steps": wandb.plot.line(
                    table,
                    x="optimizer_steps",
                    y="bits_per_byte",
                    stroke="model",
                    title="Continuous prequential code vs optimizer steps",
                ),
            }
        )

        for lr in config.learning_rates:
            pair = {
                result["tokenizer"]: result
                for result in payload["results"]
                if result["learning_rate"] == lr
            }
            bpe = pair["bpe"]
            unigram = pair["byte-unigram"]
            prefix = f"lr_{lr:g}".replace(".", "p")
            run.summary[f"{prefix}/bpe_bits_per_byte"] = bpe[
                "prequential_bits_per_byte"
            ]
            run.summary[f"{prefix}/byte_unigram_bits_per_byte"] = unigram[
                "prequential_bits_per_byte"
            ]
            run.summary[f"{prefix}/byte_unigram_minus_bpe"] = (
                unigram["prequential_bits_per_byte"] - bpe["prequential_bits_per_byte"]
            )
            for window in ("250000", "500000", "1000000", "2000000", "4000000"):
                if window in bpe["tail"] and window in unigram["tail"]:
                    run.summary[f"{prefix}/tail_{window}_bpe"] = bpe["tail"][window][
                        "bits_per_byte"
                    ]
                    run.summary[f"{prefix}/tail_{window}_byte_unigram"] = unigram[
                        "tail"
                    ][window]["bits_per_byte"]
        run.summary["metadata"] = payload["metadata"]

    print("\nFINAL")
    for lr in config.learning_rates:
        pair = {
            result["tokenizer"]: result
            for result in payload["results"]
            if result["learning_rate"] == lr
        }
        unigram = pair["byte-unigram"]
        bpe = pair["bpe"]
        uni_bpb = unigram["prequential_bits_per_byte"]
        bpe_bpb = bpe["prequential_bits_per_byte"]
        print(f"\n  lr={lr:g}")
        print(f"    byte-unigram  {uni_bpb:.6f} bits/byte")
        print(f"    bpe           {bpe_bpb:.6f} bits/byte")
        print(f"    delta         {uni_bpb - bpe_bpb:+.6f} bits/byte (Unigram - BPE)")
        for window, label in (
            ("1000000", "tail ~1 MB"),
            ("500000", "tail ~500 KB"),
            ("250000", "tail ~250 KB"),
        ):
            if window not in unigram["tail"] or window not in bpe["tail"]:
                continue
            uni_tail = unigram["tail"][window]
            bpe_tail = bpe["tail"][window]
            print(
                f"    {label:12s}  unigram={uni_tail['bits_per_byte']:.6f}  "
                f"bpe={bpe_tail['bits_per_byte']:.6f}  "
                f"delta={uni_tail['bits_per_byte'] - bpe_tail['bits_per_byte']:+.6f}"
            )
    print(f"\nWrote {output_path}; live checkpoint is {partial_path}")


if __name__ == "__main__":
    main()
