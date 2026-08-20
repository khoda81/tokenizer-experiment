# Tokenizer experiment

Controlled experiments comparing tokenizers by **continuous-stream online prequential codelength in bits per raw UTF-8 byte**.

The active work now focuses on **byte-level Unigram LM vs byte-level BPE** using the same flat-softmax Transformer. Earlier Tunstall/Bunstall experiments are retained as historical work: they answered the original token-uniformity question, but are no longer part of the default run.

## Current result

Experiment 005 found that Byte-Unigram substantially improved empirical unigram codelength over BPE while keeping almost identical bytes/token:

```text
                    bytes/token    empirical unigram bpb
Byte-Unigram           3.669              2.7484
BPE                    3.665              2.9061
```

In the matched full-stream prequential run, Byte-Unigram reached 2.2579 bpb versus BPE at 2.3118 bpb.

Experiment 006 then swept AdamW learning rate over `3e-4, 1e-3, 3e-3` using only BPE and Byte-Unigram. The best absolute learning rate for **both** tokenizers was `1e-3`:

| learning rate | Byte-Unigram | BPE | Unigram - BPE |
| ---: | ---: | ---: | ---: |
| 3e-4 | **2.286530** | 2.334183 | -0.047652 |
| 1e-3 | **2.251977** | 2.318919 | **-0.066943** |
| 3e-3 | **2.286515** | 2.368855 | -0.082341 |

At the shared optimum `1e-3`, Byte-Unigram reduces cumulative prequential codelength by about **2.89%** and still wins late in the stream:

```text
last ~1 MB:    -0.0261 bpb  (Unigram - BPE)
last ~500 KB:  -0.0192 bpb
last ~250 KB:  -0.0162 bpb
```

This rules out the simplest explanation that BPE merely needed a higher learning rate. Optimization dynamics still matter strongly: at `3e-4`, BPE eventually catches Byte-Unigram locally; at `3e-3`, both worsen but BPE is substantially less robust.

## Current question

The next experiment removes a more direct confound:

> Is Byte-Unigram better because the tokenizer saves the Transformer from spending early gradient updates learning the token marginal distribution, or does its segmentation also make the **conditional residual problem** easier?

Experiment 007 will initialize each tokenizer's output logits with a unigram prior estimated only from the ~2 MB tokenizer-fit side information. Conceptually:

```text
logits_t(x) = contextual_logits_t(x) + log p_fit(t)
```

This gives both BPE and Byte-Unigram their zero-order distribution for free and asks the Transformer to learn the contextual correction beyond that prior.

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
  001-004  historical tokenizer/uniformity experiments
  005-byte-unigram.md
  006-bpe-vs-byte-unigram-lr-sweep.md
  007-unigram-prior-initialization.md

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

Current defaults:

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
