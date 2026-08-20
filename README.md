# Tokenizer experiment

Controlled experiments comparing tokenizers by **continuous-stream online prequential codelength in bits per raw UTF-8 byte**.

The active experiment now focuses on **byte-level Unigram LM vs byte-level BPE** using the same flat-softmax Transformer. Earlier Tunstall/Bunstall experiments are retained as historical work: they answered the original token-uniformity question, but are no longer part of the default run.

## Current question

The completed five-tokenizer run found that Byte-Unigram improved held-out unigram codelength and also beat BPE in cumulative prequential bpb, while their late-stream rates nearly converged. The focused follow-up asks whether that cumulative win is primarily **faster online adaptation** rather than a persistent asymptotic representation advantage.

We therefore compare BPE and Byte-Unigram across multiple AdamW learning rates and report both:

- cumulative prequential bits/raw-byte,
- tail bits/raw-byte over approximately 250 KB, 500 KB, 1 MB, 2 MB, and 4 MB.

Tail windows are computed from the full per-update code accounting, not from sparsely logged telemetry; each requested window is moved only to the nearest real optimizer boundary.

## Layout

```text
src/tokenizer_experiment/
  model.py          reusable Transformer
  unigram.py        byte-level Unigram LM tokenizer
  tunstall.py       BPE + historical Tunstall tokenizer
  sparse_prefix.py  historical sparse-prefix / Bunstall tokenizer
  inspection.py     tokenizer diagnostics
  prequential.py    continuous-stream online prequential evaluation
  experiment.py     focused BPE/Unigram WikiText experiment

scripts/
  run_experiment.py      runnable CLI + W&B integration + LR sweep
  inspect_tokenizers.py  historical tokenizer inspection
  inspect_bunstall.py    historical Bunstall inspection

docs/experiments/
  001-005                 historical experiments
  006-bpe-vs-byte-unigram-lr-sweep.md

artifacts/               local generated outputs; gitignored

tests/
.github/workflows/ci.yml
```

## Continuous prequential protocol

The measured corpus is one continuous conditional sequence, not a collection of independent model samples. For WikiText we preserve the exact newline-joined raw text stream; row boundaries do not reset attention context.

Each tokenizer tokenizes that entire stream **once**. Optimizer updates happen near fixed raw-byte milestones, moved only to positions that are token boundaries for both tokenizers. Those update positions decide when the compressor may learn; they do not retokenize the text or reset autoregressive context.

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
uv run ruff check .
uv run pytest -q
```

Single-LR smoke test:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/run_experiment.py \
  --max-preq-mb 0.05 \
  --wandb-mode disabled
```

Focused full LR sweep:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/run_experiment.py \
  --lrs 3e-4,1e-3,3e-3 \
  --wandb-mode offline \
  --wandb-run-name bpe-unigram-lr-sweep
```

Tokenizers are trained only once; each learning rate gets a fresh Transformer with the same seed and the exact same token stream/update boundaries.

Defaults preserve the completed experiment where useful:

- model vocabulary width 4082 (4081 source-token slots + one BOS class),
- roughly first 2 MB of complete WikiText rows fit the tokenizers,
- remaining ~8.95 MB is one continuous raw stream,
- target 256 raw bytes per optimizer update,
- context 256 tokens,
- 4-layer, 256-wide Transformer,
- AdamW weight decay 0.1,
- seed 1337.

## Byte-Unigram

Byte-Unigram uses Hugging Face Tokenizers' Unigram model over ByteLevel's reversible 256-byte alphabet, with no Unicode normalization and no regex pretokenization. The trainer is asked for at most 4081 source pieces; if it underfills that target, unused model classes remain unreachable so the Transformer softmax width still exactly matches BPE.

The reported `unigram_bits_per_byte` diagnostic is computed from the **empirical emitted token frequencies on the measured stream**, not from the Unigram trainer's internal probabilities.

## Crash-safe progress

Generated state lives under gitignored `artifacts/`:

```text
artifacts/
  results.partial.json   atomically rewritten during the run
  results.json           complete result after a clean finish
  wandb/
```

`results.partial.json` is rewritten every telemetry checkpoint (100 updates by default). Every 1,000 optimizer updates, the current partial JSON is also logged as a versioned W&B progress artifact. A clean run logs the complete result.

The progress `mb` value is cumulative **raw UTF-8 megabytes encoded**, so BPE and Byte-Unigram should end at exactly the same value.

## CI

GitHub Actions installs the package with `uv`, runs Ruff, and runs CPU unit tests. CI intentionally does not download WikiText or run the GPU experiment.
