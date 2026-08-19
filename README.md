# Tokenizer experiment

Small controlled experiments comparing tokenizers by **continuous-stream online prequential codelength in bits per raw UTF-8 byte**.

The experiments compare byte-level BPE, prefix-free Tunstall-style byte phrases, and sparse-prefix ("Bunstall") tokenizers using the same flat-softmax Transformer.

> Earlier runs used either geometric block-prequential evaluation or persistent weights with the Transformer context reset at every dataset row. Those model results are legacy diagnostics, not the intended compressor-like prequential measurement.

## Layout

```text
src/tokenizer_experiment/
  model.py          reusable Transformer
  tunstall.py       BPE and Tunstall tokenizers
  sparse_prefix.py  sparse-prefix / Bunstall tokenizer
  inspection.py     tokenizer structure diagnostics
  prequential.py    continuous-stream online prequential evaluation
  experiment.py     reusable WikiText experiment orchestration

scripts/
  run_experiment.py      runnable CLI + W&B integration
  inspect_tokenizers.py  inspect BPE/Tunstall tokens and branching
  inspect_bunstall.py    inspect sparse-prefix tokenizer variants

docs/experiments/
  001-bpe-vs-tunstall-wikitext2.md       legacy block-prequential
  002-sparse-prefix-bunstall.md           tokenizer-only structural experiment
  003-legacy-block-prequential-bunstall.md
  004-online-prequential-tokenizer-comparison.md

artifacts/               local generated outputs; gitignored

tests/
.github/workflows/ci.yml
```

## Continuous prequential protocol

The measured corpus is one continuous conditional sequence, not a collection of independent model samples. For WikiText we preserve the exact newline-joined raw text stream; row boundaries do not reset attention context.

Each tokenizer tokenizes that entire stream **once**. Optimizer updates happen near fixed raw-byte milestones, moved only to positions that are token boundaries for every tokenizer. Those update positions decide when the compressor may learn; they do not retokenize the text or reset autoregressive context.

Conceptually:

```python
ids = tokenizer.encode(whole_stream)

for raw_segment in shared_update_segments:
    loss = model.nll(new_tokens, context=preceding_stream_tokens)
    prequential_bits += loss
    loss.backward()
    optimizer.step()
```

The model is initialized once, sees the stream in order exactly once, and never scores a token after learning from that token. There is no held-out score pass, no prefix retraining, and no second epoch.

The reserved special-token embedding is used only as BOS for the first token of the entire stream. There is no synthetic EOS/BOS at row or optimizer boundaries.

With context 256, large update segments are scored in subchunks of at most 128 new tokens. Each forward window therefore keeps as much preceding stream history as fits, and all subchunk gradients are accumulated before the single optimizer step for that raw segment.

## Run

```bash
uv sync --extra dev
uv run python scripts/run_experiment.py
```

Defaults:

- target 256 raw bytes per optimizer update,
- actual update positions shared by every tokenizer,
- context 256 tokens,
- one persistent model per tokenizer,
- one pass over the continuous stream,
- AdamW learning rate `1e-3`, weight decay `0.1`,
- experimental tokenizers run before established baselines.

Useful options:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/run_experiment.py \
  --wandb-mode offline \
  --wandb-run-name continuous-prequential-rerun

# Change learning cadence without changing model context.
uv run python scripts/run_experiment.py --update-bytes 128
```

## Crash-safe progress

Generated state lives under gitignored `artifacts/`:

```text
artifacts/
  results.partial.json   atomically rewritten during the run
  results.json           complete result after a clean finish
  tokenizer-inspection.json
  bunstall-inspection.json
  wandb/
```

`results.partial.json` is rewritten every telemetry checkpoint (100 updates by default). If Python receives an interruption, the file is marked `interrupted` before exit when possible.

Every 1,000 optimizer updates, the current partial JSON is also logged as a new version of the W&B Artifact `prequential-progress`. A clean run logs `prequential-results` with the complete result. These checkpoint cadences are bookkeeping only and do not affect model training or codelength.

The progress `mb` value is cumulative **raw UTF-8 megabytes encoded**, so every tokenizer should end at exactly the same MB value. Token counts differ by tokenizer.

## Inspect the learned tokenizers

Tokenizer-only inspections do not train the Transformer:

```bash
uv run python scripts/inspect_tokenizers.py
uv run python scripts/inspect_bunstall.py --mode entropy
uv run python scripts/inspect_bunstall.py --mode frequency
```

The BPE/Tunstall inspection logs the W&B Artifact `bpe-tunstall-inspection`. Bunstall inspections log `bunstall-entropy-inspection` or `bunstall-frequency-inspection`.

Inspection output includes representative emitted tokens with visible whitespace, Tunstall fanouts, BPE continuation diagnostics, and Bunstall promoted-continuation statistics.

## Default experiment

- WikiText-2 raw.
- Roughly the first 2 MB of complete rows fit the tokenizers and are shared side information.
- The remaining ~8.95 MB is one continuous raw stream.
- Requested vocabulary ~4096; the shared legal model vocabulary is 4082.
- Same 4-layer, 256-wide Transformer for every tokenizer.
- Bunstall-frequency, Bunstall-entropy, BPE, then Tunstall-boundary run in that order by default.

## CI

GitHub Actions installs the package with `uv`, runs Ruff, and runs CPU unit tests. CI intentionally does not download WikiText or run the GPU experiment.
