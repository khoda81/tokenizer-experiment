# Experiment 006 — Byte-Unigram vs BPE learning-rate sweep

## Motivation

Experiment 005 found that a byte-level Unigram LM tokenizer beat byte-level BPE on both held-out empirical unigram codelength and cumulative continuous-stream prequential codelength. On the full ~8.95 MB stream, Byte-Unigram reached about 2.2579 bpb while the matched BPE run reached about 2.3118 bpb.

However, interval analysis showed that the gap was much larger early in the stream and shrank toward the tail. This suggests that much of the cumulative advantage may come from faster online adaptation: the Unigram tokenizer has already encoded more zero-order structure, while BPE asks gradient descent to learn more of it.

## Question

Does Byte-Unigram retain its cumulative and late-stream advantage when the Transformer learning rate changes?

If increasing the learning rate substantially closes Byte-Unigram's cumulative advantage while tail rates converge, the original win is primarily an adaptation/sample-efficiency effect. If the late-stream gap persists across learning rates, that is stronger evidence for a persistent representation advantage under this finite Transformer.

## Protocol

Only two tokenizers are active:

- byte-level Unigram LM,
- byte-level BPE.

The tokenizer-fit prefix, measured continuous byte stream, model architecture, seed, weight decay, context, and raw-byte update schedule are shared. Both tokenizers are trained once. Each learning rate receives a freshly initialized Transformer with the same seed.

Default sweep:

```text
3e-4, 1e-3, 3e-3
```

Model vocabulary width remains 4082 for continuity with Experiment 005: 4081 source-token slots plus one BOS-only model class.

The experiment reports:

- cumulative prequential bits/raw-byte,
- tail bits/raw-byte over approximately 250 KB, 500 KB, 1 MB, 2 MB, and 4 MB.

Tail measurements use the nearest actual optimizer boundary and are computed from every update's cumulative code, independently of telemetry sampling.

## Run

```bash
HF_HUB_OFFLINE=1 uv run python scripts/run_experiment.py \
  --lrs 3e-4,1e-3,3e-3 \
  --wandb-mode offline \
  --wandb-run-name bpe-unigram-lr-sweep
```

## Status

Pending focused rerun.
