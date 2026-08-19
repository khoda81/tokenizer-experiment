# Tokenizer experiment

Small controlled experiments comparing tokenizers by **block-prequential codelength in bits per raw UTF-8 byte**.

The first experiment compares byte-level BPE with a prefix-free Tunstall-style byte phrase vocabulary using the same flat-softmax Transformer.

## Layout

```text
src/tokenizer_experiment/
  model.py          reusable Transformer
  tunstall.py       BPE and Tunstall tokenizers
  inspection.py     tokenizer structure diagnostics
  prequential.py    one-pass block-prequential evaluation
  experiment.py     reusable WikiText experiment orchestration

scripts/
  run_experiment.py      runnable CLI + W&B integration
  inspect_tokenizers.py  inspect learned tokens and branching structure

docs/experiments/
  001-bpe-vs-tunstall-wikitext2.md

tests/
.github/workflows/ci.yml
```

Reusable logic belongs under `src/tokenizer_experiment`. Scripts should be thin entry points around that package.

## Run

```bash
uv sync --extra dev
uv run python scripts/run_experiment.py
```

W&B logging is enabled by default under project `tokenizer-experiment`. The run logs each prequential checkpoint live and stores a final table plus curves for cumulative bits/byte against both raw bytes and optimizer steps.

Useful W&B options:

```bash
# Explicit project / run name
uv run python scripts/run_experiment.py \
  --wandb-project tokenizer-experiment \
  --wandb-run-name bpe-vs-tunstall-seed-1337

# Keep W&B data local
uv run python scripts/run_experiment.py --wandb-mode offline

# Disable W&B entirely; results.json is still written
uv run python scripts/run_experiment.py --wandb-mode disabled
```

## Inspect the learned tokenizers

The inspection run trains only the tokenizers, not the Transformer:

```bash
uv run python scripts/inspect_tokenizers.py
```

It prints and writes `tokenizer-inspection.json` containing:

- the most frequent emitted Tunstall and BPE tokens,
- every expanded Tunstall prefix and its observed 256-way next-byte distribution,
- the next-byte entropy and effective branching factor of each Tunstall expansion,
- learned BPE merges viewed as approximate binary continuation tests `A+B` versus `A+[not B]`,
- the binary entropy of those BPE splits and summary statistics over supported merges.

This is intended to test the hypothesis that BPE spends one vocabulary slot at a time on sparse binary refinements while a byte-level Tunstall expansion spends 255 extra leaves at once.

## Default experiment

- WikiText-2 raw.
- First 2 MB fit both tokenizers and are treated as shared side information.
- All remaining train bytes form the candidate prequential stream.
- Requested vocabulary ~4096; the byte-only Tunstall tree plus separate EOS snaps this to 4082.
- Checkpoints: `1%, 2%, 4%, 8%, 16%, 32%, 64%, 100%`.
- Same 4-layer, 256-wide Transformer for both tokenizers.
- Context 256 tokens.
- Each observed prefix is trained from scratch for **exactly one pass**. There is no epoch parameter.
- First block uses a uniform token code; later blocks use causal NLL from the model trained on the preceding raw-byte prefix.

Tunstall-safe boundaries are used for both tokenizers so every scored block contains exactly the same raw bytes.

## Outputs

`results.json` is the machine-readable source of truth for each run. It contains metadata, stage records, final prequential code, and a compact `code_curve` containing:

- raw bytes encoded,
- cumulative bits and bits/byte,
- training bytes and tokens,
- per-model optimizer steps,
- cumulative optimizer steps,
- cumulative train + score wall time.

W&B mirrors this trace for interactive comparisons; it does not replace the JSON result.

## Experiments

See [`docs/experiments/001-bpe-vs-tunstall-wikitext2.md`](docs/experiments/001-bpe-vs-tunstall-wikitext2.md) for the first recorded run and its stage-by-stage results.

## CI

GitHub Actions installs the package with `uv`, runs Ruff, and runs the CPU unit tests. CI intentionally does **not** download WikiText or run the GPU experiment.
