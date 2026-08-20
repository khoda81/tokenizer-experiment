# Experiment 005 — Byte-Unigram vs BPE

Date: 2026-08-20  
Status: **complete**

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

## Protocol

The tokenizer is fit on the same ~2 MB side-information prefix as the earlier tokenizers. The measured stream is the remaining ~8.95 MB of newline-joined WikiText-2 training text.

All tokenizers use the same 4082-class Transformer and one-pass continuous-stream online prequential protocol. Optimizer boundaries are shared raw-byte positions. In this five-tokenizer run, including Byte-Unigram changed the exact common-boundary schedule slightly relative to Experiment 004, so BPE and all other baselines were rerun in the same experiment.

## Tokenizer-only results

On the full measured stream:

| tokenizer | bytes/token | H(T)/log2(V) | empirical unigram bpb | used classes |
| --- | ---: | ---: | ---: | ---: |
| Byte-Unigram | 3.669 | 0.8407 | **2.7484** | 3782/4082 |
| BPE | 3.665 | 0.8880 | 2.9061 | 3948/4082 |
| Bunstall-frequency | 3.170 | 0.9116 | 3.4497 | 3770/4082 |
| Bunstall-entropy | 3.062 | 0.9557 | 3.7435 | 4017/4082 |
| Tunstall-boundary | 1.883 | 0.6476 | 4.1263 | 948/4082 |

Byte-Unigram and BPE have almost identical bytes/token, but Byte-Unigram reduces empirical zero-order codelength by about **0.1577 bits/raw-byte**. The improvement therefore comes primarily from a lower-entropy emitted token distribution rather than from packing more raw bytes into each Transformer token.

## Prequential results

Matched full-stream results under the same run and update schedule:

| tokenizer | prequential bpb | delta vs BPE |
| --- | ---: | ---: |
| Byte-Unigram | **2.257925** | **-0.053858** |
| BPE | 2.311782 | baseline |
| Bunstall-frequency | 2.512061 | +0.200279 |
| Bunstall-entropy | 2.598281 | +0.286498 |

The full Tunstall run was not needed for the focused follow-up after the BPE/Unigram comparison was established.

Byte-Unigram beats BPE by about **0.0539 bpb**, or roughly **2.3%** of BPE's prequential codelength.

## Interpretation

The zero-order advantage is larger than the final Transformer advantage:

```text
empirical unigram gap:      ~0.1577 bpb
final prequential gap:      ~0.0539 bpb
```

So the Transformer learns away a substantial fraction of the extra statistical structure that BPE leaves in its token stream, but not all of it within this finite one-pass online regime.

Interval analysis showed that the Byte-Unigram advantage was very large early in training and shrank strongly toward the tail. This motivated Experiment 006: sweep the Transformer learning rate to distinguish a generic optimization/adaptation-speed effect from a persistent late-stream advantage.

## Important metric caveat

The reported `unigram_bits_per_byte` is a **retrospective empirical diagnostic** computed from emitted token frequencies on the measured stream:

```text
H(T) / mean_bytes_per_token
```

It is not a held-out codelength under token probabilities estimated only from the tokenizer-fit prefix. That distinction matters for any later experiment that explicitly supplies the model with a unigram prior.

## Outcome

Both primary hypotheses were supported in this run:

1. Byte-Unigram substantially improved empirical unigram bpb over BPE.
2. The improvement translated into lower Transformer prequential bpb.

The result shifted the active project from broad tokenizer-family comparison to a focused BPE-vs-Byte-Unigram investigation of optimization and sample efficiency.
