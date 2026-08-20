# Experiment 005 — Byte-Unigram vs BPE

Date: 2026-08-20  
Status: **planned / implemented, awaiting run**

## Motivation

Experiment 004 found that held-out empirical unigram bits/raw-byte tracked final continuous prequential performance much more closely than token-marginal uniformity. The ordering was identical for BPE, Bunstall-frequency, Bunstall-entropy, and Tunstall-boundary.

This experiment asks the obvious follow-up:

> If zero-context source coding is the strongest tokenizer-level predictor we have observed, what happens when the tokenizer is explicitly trained as a unigram language model rather than by greedy BPE merges or prefix refinements?

## Tokenizer

`ByteUnigramTokenizer` wraps Hugging Face Tokenizers' Unigram model/trainer with ByteLevel preprocessing.

Important constraints:

- no Unicode normalization,
- `ByteLevel(add_prefix_space=False, use_regex=False)`,
- full 256-symbol ByteLevel alphabet is retained,
- no unknown source token is required,
- 4081 source pieces at the default shared model vocabulary,
- model class 4081 is reserved externally as BOS and is not part of the source-token trainer,
- default maximum source-piece length: 16 bytes.

The Unigram trainer uses its standard EM/pruning likelihood objective. We do **not** report the trainer's own likelihood as the comparison metric.

## Primary tokenizer-only test

On the same held-out continuous stream used by Experiment 004, compute from the deterministic emitted token IDs:

```text
bytes/token
H(T) / log2(V)
unigram bits/raw-byte = H(T) / bytes_per_token
vocabulary utilization
```

The key baseline from Experiment 004 is BPE:

```text
BPE held-out unigram bits/raw-byte = 2.9061
```

Byte-Unigram runs first in the experiment. If it cannot improve on BPE's held-out unigram bpb, the tokenizer-only result is already informative and the Transformer run may be stopped early if desired.

## Model test

If Byte-Unigram is competitive on held-out unigram bpb, compare it under the exact same continuous-stream prequential protocol:

- same ~8.95 MB raw stream,
- same shared raw-byte optimizer boundaries,
- target 256 raw bytes/update,
- same 4082-class flat-softmax Transformer,
- same context and optimizer hyperparameters,
- same one-pass prequential code calculation.

Execution order:

1. Byte-Unigram,
2. Bunstall-frequency,
3. Bunstall-entropy,
4. BPE,
5. Tunstall-boundary.

## Hypotheses

The strongest version of the emerging hypothesis predicts:

1. Byte-Unigram improves held-out unigram bpb over BPE, and
2. that improvement translates into lower final Transformer prequential bpb.

Possible negative results are also useful:

- **better fit-corpus likelihood but worse held-out unigram bpb:** vocabulary/segmentation overfit or objective mismatch;
- **better held-out unigram bpb but worse Transformer bpb:** zero-order compression is not sufficient; contextual learnability matters independently;
- **BPE still wins both:** BPE's greedy merge construction is already a surprisingly strong approximation to the useful zero-order objective at this scale.

## Run

Smoke test:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/run_experiment.py \
  --max-preq-mb 0.05 \
  --wandb-mode disabled
```

Full run:

```bash
HF_HUB_OFFLINE=1 uv run python scripts/run_experiment.py \
  --wandb-mode offline \
  --wandb-run-name byte-unigram
```

## Results

Pending.
