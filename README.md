# Tokenizer experiment

Small controlled experiments comparing tokenizers by **online prequential codelength in bits per raw UTF-8 byte**.

The experiments compare byte-level BPE, prefix-free Tunstall-style byte phrases, and sparse-prefix ("Bunstall") tokenizers using the same flat-softmax Transformer.

> Experiments 001 and 003 used an earlier geometric **block-prequential** evaluator. Those model results are retained as legacy diagnostics but are not the intended online-prequential measurement.

## Layout

```text
src/tokenizer_experiment/
  model.py          reusable Transformer
  tunstall.py       BPE and Tunstall tokenizers
  sparse_prefix.py  sparse-prefix / Bunstall tokenizer
  inspection.py     tokenizer structure diagnostics
  prequential.py    online datum-by-datum prequential evaluation
  experiment.py     reusable WikiText experiment orchestration

scripts/
  run_experiment.py      runnable CLI + W&B integration
  inspect_tokenizers.py  inspect BPE/Tunstall tokens and branching
  inspect_bunstall.py    inspect sparse-prefix tokenizer variants

docs/experiments/
  001-bpe-vs-tunstall-wikitext2.md       legacy block-prequential
  002-sparse-prefix-bunstall.md           tokenizer-only structural experiment
  003-legacy-block-prequential-bunstall.md

artifacts/               local generated outputs; gitignored

tests/
.github/workflows/ci.yml
```

Reusable logic belongs under `src/tokenizer_experiment`. Scripts should be thin entry points around that package.

## Correct online-prequential protocol

The model is initialized once and walks through the stream exactly once:

```python
for datum in stream:
    loss = model.loss(datum)   # probability before learning this datum
    prequential_bits += loss
    loss.backward()
    optimizer.step()           # exactly one update for this datum
```

There is no held-out scoring pass, no geometric blocking, no retraining from scratch, and no second epoch. Batch size is one **raw datum**.

The default datums are as fine-grained as possible while remaining identical for every tokenizer: start from individual WikiText dataset rows and greedily merge adjacent rows only when necessary for a prefix-free Tunstall phrase to terminate at the datum boundary.

Each datum is modeled as:

```text
<EOS-as-BOS> content tokens <EOS>
```

so the first content token is predicted by the model rather than charged an artificial uniform code. If an unusually long datum exceeds the 256-token model context, its loss is accumulated over context-sized chunks and the optimizer still steps only once after the complete datum has been scored.

## Run

```bash
uv sync --extra dev
uv run python scripts/run_experiment.py
```

Defaults include:

- one persistent model per tokenizer,
- batch size one datum,
- exactly one pass over the ordered stream,
- AdamW learning rate `1e-3`, deliberately aggressive for online learning,
- W&B telemetry every 100 datums; logging frequency does not affect the code.

Useful W&B options:

```bash
# Explicit project / run name
uv run python scripts/run_experiment.py \
  --wandb-project tokenizer-experiment \
  --wandb-run-name online-bunstall-uniformity

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

## Default experiment

- WikiText-2 raw.
- Roughly the first 2 MB of **complete dataset rows** fit the tokenizers and are treated as shared side information.
- All remaining complete rows form the candidate online stream.
- Requested vocabulary ~4096; the byte-only Tunstall tree plus separate EOS snaps this to 4082.
- Same 4-layer, 256-wide Transformer for every tokenizer.
- Context 256 tokens.
- One model initialization, one ordered pass, one optimizer update per raw datum.
- Learning rate `1e-3` by default.
- BPE, Tunstall-boundary, Bunstall-entropy, and Bunstall-frequency are compared on exactly the same raw datums.

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

`prequential-results` contains the machine-readable online result with protocol metadata, final prequential code, tokenizer diagnostics, datum counts, and a compact `code_curve` sampled every logging interval with:

- datum / optimizer step,
- cumulative raw bytes and tokens,
- cumulative bits and bits/byte,
- current datum bits/byte,
- elapsed wall time.

## Experiments

See `docs/experiments/` for recorded hypotheses, setups, results, and explicit legacy labels.

## CI

GitHub Actions installs the package with `uv`, runs Ruff, and runs the CPU unit tests. CI intentionally does **not** download WikiText or run the GPU experiment.
