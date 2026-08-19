# Experiment 004 — Continuous-stream prequential tokenizer comparison

Date: 2026-08-19  
Status: **planned / corrected rerun**

## Question

With the same flat-softmax Transformer and vocabulary size, how do BPE, prefix-free Tunstall, Bunstall-entropy, and Bunstall-frequency compare when the model is treated as one adaptive compressor over one continuous text stream?

A secondary question is whether a more uniform emitted-token marginal makes the conditional model easier to learn online, independently of zero-order tokenizer compression.

## Protocol correction

Two earlier model-evaluation protocols are now considered legacy for this question:

1. geometric block-prequential runs that repeatedly retrained fresh models on prefixes, and
2. the transitional online implementation at commit `a975989`, which preserved model weights but reset Transformer context at every WikiText row.

The intended object is one continuous conditional sequence. WikiText rows remain separated by their literal newline bytes, but row boundaries have no model-level meaning and do not reset context.

Each tokenizer tokenizes the entire measured stream once. The model then walks through that token stream exactly once. Optimizer update positions are shared raw-byte milestones, moved to byte positions that are token boundaries for every tokenizer.

Conceptually:

```python
ids = tokenizer.encode(whole_stream)

for update_segment in shared_raw_segments:
    loss = nll(new_tokens | preceding_stream_context)
    prequential_bits += loss
    loss.backward()
    optimizer.step()
```

Every probability charged to an update is produced by the model state before that update. There is no separate score pass, no retokenization at update boundaries, no context reset at update boundaries, and no second epoch.

## Stream construction

- WikiText-2 raw train split.
- Roughly the first 2 MB of complete dataset rows fit all tokenizers and are treated as shared side information.
- The remaining ~8.95 MB is concatenated exactly as the newline-joined raw WikiText stream.
- Only a possible final partial Tunstall phrase is dropped so the finite measured stream ends on a token boundary.
- Dataset rows do not define optimizer batches or attention resets.

## Shared optimizer cadence

Default target cadence: **256 raw UTF-8 bytes per optimizer update**.

For each tokenizer, the complete continuous tokenization is converted to raw-byte token-boundary offsets. We intersect those boundary sets. For raw milestones `256, 512, 768, ...`, the update moves to the nearest following common token boundary.

Therefore:

- all tokenizers update after the same raw text,
- tokenization is independent of update placement,
- optimizer-step count is shared,
- token counts per update can differ.

The attention context is a separate quantity: 256 tokens. If an update contains many new tokens, it is scored in subchunks of at most 128 new tokens so each forward reserves substantial room for preceding stream context. All subchunk gradients accumulate before the one optimizer step for that raw segment.

## Training

- One model initialization per tokenizer.
- One ordered pass over the stream.
- AdamW learning rate `1e-3` initial aggressive default.
- Weight decay `0.1`.
- Transformer: 4 layers, width 256, 4 heads, context 256 tokens.
- The reserved special-token embedding is used only as BOS for the first token of the entire stream.

## Tokenizers / execution order

Experimental variants run before established controls:

1. Bunstall-frequency,
2. Bunstall-entropy,
3. byte-level BPE,
4. Tunstall-boundary.

Tokenizer diagnostics include bytes/token, marginal token entropy, `H(T)/log2(V)`, unigram bits/raw-byte, and vocabulary utilization.

## Crash-safe logging

- `artifacts/results.partial.json` is atomically rewritten every 100 updates by default.
- Every 1,000 updates the partial snapshot is logged as a version of W&B Artifact `prequential-progress`.
- A clean finish writes `artifacts/results.json` and logs `prequential-results`.
- Progress/artifact cadence is telemetry only and does not affect coding or training.

## Run

```bash
HF_HUB_OFFLINE=1 uv run python scripts/run_experiment.py \
  --wandb-mode offline \
  --wandb-run-name continuous-prequential-rerun
```

## Results

Pending corrected rerun.
