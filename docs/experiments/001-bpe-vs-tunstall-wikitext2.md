# Experiment 001 — BPE vs prefix-free Tunstall on WikiText-2

> **LEGACY / protocol correction (2026-08-19):** this experiment used a geometric **block-prequential** evaluator: fresh models were repeatedly trained on preceding prefixes and then scored on separate future blocks. That is not the online datum-by-datum prequential protocol intended for this project. The numerical model comparison below is retained as historical evidence only and will be repeated. Tokenizer-only diagnostics remain valid.

Date: 2026-08-19  
Baseline commit: `8f0db6c`  
Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`

## Question

With the same flat-softmax Transformer and approximately the same vocabulary size, does a prefix-free byte-phrase vocabulary make the sequential learning problem easier than byte-level BPE when measured by block-prequential codelength?

The comparison metric is **bits per raw UTF-8 byte**, not token perplexity.

## Setup

- WikiText-2 train split: 10.95 MB UTF-8.
- First 2.00 MB used to fit both tokenizers and treated as shared side information.
- Remaining 8.95 MB used as the prequential stream.
- Requested vocabulary: 4096.
- Exact shared model vocabulary: 4082.
  - Tunstall: 4081 prefix-free byte-phrase leaves + one separate EOS token.
  - BPE: 4082 vocabulary entries including its separate EOS token.
- Tunstall mode: `boundary`.
- Block-prequential cuts: 1%, 2%, 4%, 8%, 16%, 32%, 64%, 100%, moved slightly backward when necessary so every cut is both a UTF-8 boundary and a completed Tunstall phrase.
- Transformer: 4 layers, width 256, 4 heads, context 256 tokens.
- AdamW, learning rate `3e-4`, weight decay `0.1`, batch size 16.
- Every block-prequential prefix was trained **from scratch for exactly one pass**.
- The first block was encoded with a uniform token code.

Aligned byte cuts were:

```text
0.090 MB, 0.179 MB, 0.358 MB, 0.716 MB,
1.432 MB, 2.864 MB, 5.729 MB, 8.952 MB
```

## Tokenizer diagnostics

| tokenizer | bytes/token | H(T) / log2(V) | vocab used |
|---|---:|---:|---:|
| Tunstall | 1.883 | 0.6476 | 948 / 4082 |
| BPE | 3.665 | 0.8880 | 3948 / 4082 |

The Tunstall tree had 4081 phrase leaves but only 15 internal-node expansions, because a full byte-tree expansion replaces one leaf with 256 children and therefore costs 255 additional leaves. Its longest phrase was only 3 bytes.

This means the current Tunstall construction does **not** achieve the motivating goal of a nearly uniform marginal token distribution at this vocabulary size. That tokenizer result remains valid.

## Legacy block-prequential results

### BPE

| stage | train bytes | score bytes | optimizer steps | block bits/byte | cumulative bits/byte |
|---:|---:|---:|---:|---:|---:|
| uniform | 0 | 0.090 MB | 0 | 3.2476 | 3.2476 |
| 1 | 0.09 MB | 0.09 MB | 6 | 3.0306 | 3.1391 |
| 2 | 0.18 MB | 0.18 MB | 12 | 2.9867 | 3.0629 |
| 3 | 0.36 MB | 0.36 MB | 24 | 2.9428 | 3.0029 |
| 4 | 0.72 MB | 0.72 MB | 47 | 2.9085 | 2.9557 |
| 5 | 1.43 MB | 1.43 MB | 95 | 2.9261 | 2.9409 |
| 6 | 2.86 MB | 2.86 MB | 190 | 2.9198 | 2.9304 |
| 7 | 5.73 MB | 3.22 MB | 382 | 2.6559 | **2.8316** |

### Tunstall

| stage | train bytes | score bytes | optimizer steps | block bits/byte | cumulative bits/byte |
|---:|---:|---:|---:|---:|---:|
| uniform | 0 | 0.090 MB | 0 | 6.3689 | 6.3689 |
| 1 | 0.09 MB | 0.09 MB | 12 | 5.3023 | 5.8356 |
| 2 | 0.18 MB | 0.18 MB | 24 | 4.6066 | 5.2211 |
| 3 | 0.36 MB | 0.36 MB | 47 | 4.2029 | 4.7120 |
| 4 | 0.72 MB | 0.72 MB | 93 | 4.1140 | 4.4130 |
| 5 | 1.43 MB | 1.43 MB | 186 | 4.0415 | 4.2272 |
| 6 | 2.86 MB | 2.86 MB | 372 | 3.4005 | 3.8139 |
| 7 | 5.73 MB | 3.22 MB | 744 | 2.9151 | **3.4903** |

Historical result:

```text
BPE                  2.831569 bits/byte
Tunstall-boundary    3.490299 bits/byte
Tunstall - BPE      +0.658730 bits/byte
```

These values describe the legacy block-prequential setup only. They must not be cited as the project's online prequential result.

## What remains useful

The structural tokenizer findings remain useful:

- BPE produces about half as many tokens per raw byte, so a 256-token context covers roughly twice as many bytes.
- BPE is much closer to marginally uniform than the 256-way Tunstall construction.
- Full byte-level Tunstall expansion is extremely expensive at ~4k vocabulary because each expansion consumes 255 additional slots.

The model comparison itself is being repeated using one persistent model and the corrected online loop: score one raw datum, update once from that same loss, and never revisit it.
