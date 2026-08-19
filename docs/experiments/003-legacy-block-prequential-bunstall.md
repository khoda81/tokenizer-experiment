# Experiment 003 — Legacy block-prequential BPE / Tunstall / Bunstall run

Date: 2026-08-19  
Status: **LEGACY — invalid for the intended online-prequential question**

## Why this result is legacy

This run used the repository's original `run_block_prequential` protocol. It divided the prequential stream into geometric blocks, repeatedly initialized a fresh model, trained that model on the complete preceding prefix, and then made a separate inference pass over the next held-out block.

That is a legitimate **block-prequential** experiment, but it is not the online prequential code intended in this project.

The intended protocol is instead:

```python
for datum in stream:
    loss = model.loss(datum)   # probability before learning datum
    prequential_bits += loss
    loss.backward()
    optimizer.step()           # learn datum exactly once
```

There is one model, one ordered pass through the stream, batch size one datum, and no second epoch or separate scoring pass.

The numbers below are retained only as historical diagnostics and must not be compared as if they were results from the corrected online protocol.

## Tokenizer diagnostics

These tokenizer-only measurements are still useful because they do not depend on the mistaken prequential evaluator.

| tokenizer | bytes/token | H(T) / log2(V) | unigram bpb | vocab used |
|---|---:|---:|---:|---:|
| BPE | 3.665 | 0.8880 | 2.9060 | 3948 / 4082 |
| Tunstall-boundary | 1.883 | 0.6476 | 4.1263 | 948 / 4082 |
| Bunstall-entropy | 3.062 | 0.9557 | 3.7435 | 4017 / 4082 |
| Bunstall-frequency | 3.170 | 0.9114 | 3.4492 | 3770 / 4082 |

## Legacy block-prequential result

| tokenizer | legacy block-prequential bpb |
|---|---:|
| BPE | 2.831569 |
| Bunstall-frequency | 3.211263 |
| Bunstall-entropy | 3.379674 |
| Tunstall-boundary | 3.490299 |

Again: these numbers answer a different block-prequential question. The experiment is being repeated from scratch using the corrected online datum-by-datum protocol.
