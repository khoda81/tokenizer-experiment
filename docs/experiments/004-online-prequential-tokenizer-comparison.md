# Experiment 004 — Online prequential tokenizer comparison

Date: 2026-08-19  
Status: **planned / rerun of legacy model comparisons with corrected protocol**

## Question

With the same flat-softmax Transformer and vocabulary size, how do BPE, prefix-free Tunstall, Bunstall-entropy, and Bunstall-frequency compare when the model is evaluated by true online prequential code?

A secondary question is whether a more uniform emitted-token marginal makes the conditional model easier to learn online, independently of zero-order tokenizer compression.

## Protocol correction

Experiments 001 and 003 used geometric block-prequential coding. This experiment uses one persistent model and one ordered pass:

```python
for datum in stream:
    loss = model.loss(datum)
    prequential_bits += loss
    loss.backward()
    optimizer.step()
```

The loss is measured before the datum is learned. The exact same loss supplies the gradients for the one update on that datum. There is no separate scoring pass and no retraining from scratch.

## Datum policy

The goal is to maximize the number of natural raw datums while using identical byte ranges for every tokenizer.

1. Start from individual WikiText dataset rows.
2. Fit tokenizers on roughly the first 2 MB of complete rows; this is shared side information.
3. On the remaining rows, close a datum at every row boundary where the prefix-free Tunstall parser has returned to its root.
4. Merge adjacent rows only when a Tunstall phrase straddles their boundary.
5. Drop only a final incomplete group if the dataset ends inside a Tunstall phrase.

This greedily produces the maximum number of row-aligned datums allowed by the current Tunstall vocabulary. Every tokenizer sees exactly the same raw datums.

Each datum is scored as:

```text
<EOS-as-BOS> content tokens
```

The reserved EOS embedding is reused only as fixed start-of-datum context. The row/datum boundary itself is known side information, so it is **not** charged as an additional synthetic EOS target. The literal raw row separator is already present as its newline byte.

## Training

- Batch size: **1 raw datum**.
- Passes over the stream: **1**.
- A second epoch is intentionally undefined for this prequential measurement because a datum cannot be scored again as unseen after it has been learned.
- AdamW learning rate: `1e-3` initial aggressive default.
- Weight decay: `0.1`.
- Transformer: 4 layers, width 256, 4 heads, context 256 tokens.
- One optimizer step per datum.
- If a datum exceeds the token context, all of its chunks are scored and backpropagated before the single optimizer step.

The maximum stable learning rate is an optimization hyperparameter rather than part of the prequential definition. It should be tuned using only side information / a separate tuning stream, never by looking ahead into the measured online stream.

## Tokenizers

Experimental variants run first so an obviously unpromising new tokenizer can be stopped before spending time revalidating established baselines.

Current order:

1. Bunstall-frequency,
2. Bunstall-entropy,
3. byte-level BPE,
4. Tunstall-boundary.

Tokenizer diagnostics include bytes/token, marginal token entropy, `H(T)/log2(V)`, unigram bits/raw-byte, and vocabulary utilization.

## Logging

W&B logging every N datums is telemetry only and does not alter datum boundaries, model updates, or code length. The full result is logged as the versioned W&B Artifact `prequential-results`.

## Run

```bash
HF_HUB_OFFLINE=1 uv run python scripts/run_experiment.py \
  --wandb-mode offline \
  --wandb-run-name online-prequential-rerun
```

## Results

Pending rerun.
