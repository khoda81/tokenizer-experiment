# Tokenizer experiment

Small controlled experiments comparing tokenizers by **block-prequential codelength in bits per raw UTF-8 byte**.

The experiments compare byte-level BPE, prefix-free Tunstall-style byte phrases, and sparse-prefix ("Bunstall") tokenizers using the same flat-softmax Transformer.

## Layout

```text
src/tokenizer_experiment/
  model.py          reusable Transformer
  tunstall.py       BPE and Tunstall tokenizers
  sparse_prefix.py  sparse-prefix / Bunstall tokenizer
  inspection.py     tokenizer structure diagnostics
  prequential.py    one-pass block-prequential evaluation
  experiment.py     reusable WikiText experiment orchestration

scripts/
  run_experiment.py      runnable CLI + W&B integration
  inspect_tokenizers.py  inspect BPE/Tunstall tokens and branching
  inspect_bunstall.py    inspect sparse-prefix tokenizer variants

docs/experiments/
  001-bpe-vs-tunstall-wikitext2.md
  002-sparse-prefix-bunstall.md

artifacts/               local generated outputs; gitignored

tests/
.github/workflows/ci.yml
```

Reusable logic belongs under `src/tokenizer_experiment`. Scripts should be thin entry points around that package.

## Run

```bash
uv sync --extra dev
uv run python scripts/run_experiment.py
```

W&B logging is enabled by default under project `tokenizer-experiment`. The run logs each prequential checkpoint live, stores comparison tables and curves, and logs the full result JSON as the versioned W&B Artifact `prequential-results`.

Useful W&B options:

```bash
# Explicit project / run name
uv run python scripts/run_experiment.py \
  --wandb-project tokenizer-experiment \
  --wandb-run-name bunstall-uniformity-test

# Keep W&B data local for later syncing
uv run python scripts/run_experiment.py --wandb-mode offline

# Disable W&B entirely; the local staged JSON is still written
uv run python scripts/run_experiment.py --wandb-mode disabled
```

When the Hugging Face dataset is already cached and network access is flaky, skip Hub requests entirely:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/run_experiment.py --wandb-mode offline
```

## Inspect the learned tokenizers

Tokenizer-only inspections do not train the Transformer:

```bash
uv run python scripts/inspect_tokenizers.py
uv run python scripts/inspect_bunstall.py --mode entropy
uv run python scripts/inspect_bunstall.py --mode frequency
```

The BPE/Tunstall inspection logs the W&B Artifact `bpe-tunstall-inspection`. Bunstall inspections log `bunstall-entropy-inspection` or `bunstall-frequency-inspection`.

Inspection output includes:

- the most frequent emitted tokens, with visible whitespace glyphs,
- expanded Tunstall prefixes and their observed 256-way continuation distributions,
- next-byte entropy and effective branching factor,
- learned BPE merges viewed as approximate binary continuation tests,
- Bunstall promoted continuations and their binary split statistics.

This tests the hypothesis that BPE and sparse-prefix tokenizers can spend one vocabulary slot at a time on useful refinements while a byte-level Tunstall expansion must spend 255 extra leaves at once.

## Default experiment

- WikiText-2 raw.
- First 2 MB fit the tokenizers and are treated as shared side information.
- All remaining train bytes form the candidate prequential stream.
- Requested vocabulary ~4096; the byte-only Tunstall tree plus separate EOS snaps this to 4082.
- Checkpoints: `1%, 2%, 4%, 8%, 16%, 32%, 64%, 100%`.
- Same 4-layer, 256-wide Transformer for every tokenizer.
- Context 256 tokens.
- Each observed prefix is trained from scratch for **exactly one pass**. There is no epoch parameter.
- First block uses a uniform token code; later blocks use causal NLL from the model trained on the preceding raw-byte prefix.

The current experiment preserves the original Tunstall-safe raw-byte checkpoint policy for comparability with Experiment 001.

## Outputs and W&B Artifacts

Generated local state is staged under `artifacts/`, which is gitignored:

```text
artifacts/
  results.json
  tokenizer-inspection.json
  bunstall-inspection.json
  wandb/
```

The local JSON files are useful for immediate inspection, but **W&B Artifacts are the persistent experiment record**. Repeated logs of the same artifact collection create versioned outputs associated with their producing runs.

`prequential-results` contains the machine-readable experiment result with metadata, stage records, final prequential code, and a compact `code_curve` containing:

- raw bytes encoded,
- cumulative bits and bits/byte,
- training bytes and tokens,
- per-model optimizer steps,
- cumulative retraining work,
- cumulative train + score wall time.

## Experiments

See `docs/experiments/` for recorded hypotheses, setups, and results.

## CI

GitHub Actions installs the package with `uv`, runs Ruff, and runs the CPU unit tests. CI intentionally does **not** download WikiText or run the GPU experiment.
