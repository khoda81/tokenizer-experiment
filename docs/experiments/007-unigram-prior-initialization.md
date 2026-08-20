# Experiment 007 — Unigram-prior initialization

Date: 2026-08-21  
Status: **planned**

## Motivation

Experiments 005 and 006 established that Byte-Unigram beats BPE in continuous-stream online prequential codelength under this finite Transformer.

The learning-rate sweep ruled out the simplest optimizer explanation: `1e-3` is the best absolute learning rate for both tokenizers, and Byte-Unigram still wins cumulative and late-stream codelength there.

However, prequential coding charges the model for learning each tokenizer's marginal token frequencies from scratch. Byte-Unigram was explicitly optimized as a unigram language model, so it may win primarily because its emitted zero-order distribution is easier to learn online.

This experiment removes that factor directly.

## Question

If BPE and Byte-Unigram both receive their tokenizer-specific unigram distribution from side information before the measured stream begins, does Byte-Unigram still produce a lower residual conditional codelength?

## Intervention

Estimate token probabilities separately for each tokenizer using **only the tokenizer-fit side-information prefix**.

With token counts `n_t`, total fit tokens `N`, and smoothing parameter `alpha`, define

```text
p_fit(t) = (n_t + alpha) / (N + alpha * V)
```

Use Jeffreys smoothing by default:

```text
alpha = 0.5
```

Initialize a per-token additive output bias with

```text
b_t = log p_fit(t)
```

and predict with

```text
logits_t(x) = contextual_logits_t(x) + b_t
```

The prior must be estimated independently for BPE and Byte-Unigram from the same raw fit prefix. No measured-stream token counts may be used.

## Why an output bias

The current Transformer ties input embeddings and output weights and has no output-head bias. Encoding the unigram prior into the tied embedding matrix would change both token representations and contextual computation.

A separate additive logit bias is the clean intervention: it supplies zero-order log odds without modifying token embeddings.

At initialization the contextual logits are near zero, so predictions begin close to `p_fit(t)`. Training then learns residual conditional information beyond the marginal prior.

Conceptually:

```text
log p(t | context)
  = log p_fit(t)
  + contextual correction
  - normalization
```

## Primary comparison

At the best learning rate from Experiment 006 (`1e-3`), compare four conditions:

| tokenizer | random zero-order initialization | fit-unigram prior |
| --- | --- | --- |
| BPE | baseline from Experiment 006 | prior-initialized BPE |
| Byte-Unigram | baseline from Experiment 006 | prior-initialized Byte-Unigram |

The most important comparison is **prior-initialized BPE vs prior-initialized Byte-Unigram**.

## Measurements

Report:

- cumulative prequential bits/raw-byte,
- local / tail codelength over ~250 KB, 500 KB, 1 MB, 2 MB, and 4 MB,
- initial zero-context cross-entropy on the first measured updates,
- cumulative improvement from supplying the prior for each tokenizer.

Also report the fit-prior held-out zero-order cross-entropy:

```text
-sum_test log2 p_fit(t) / raw_test_bytes
```

This is distinct from the retrospective empirical `unigram_bits_per_byte` diagnostic used in Experiments 004-006 and measures actual generalization of the side-information marginal model.

## Hypotheses

### H1 — marginal-learning explanation

If prior initialization collapses most of the BPE-vs-Unigram gap, Byte-Unigram's main benefit is sample efficiency: the tokenizer saves gradient descent from rediscovering zero-order token frequencies.

### H2 — residual conditional advantage

If Byte-Unigram still wins materially after both models receive their marginals for free, the segmentation itself makes the finite Transformer's conditional prediction problem easier.

### H3 — BPE residual advantage

If BPE overtakes Byte-Unigram after prior initialization, BPE may leave more contextual structure available to the neural model while its previous disadvantage came mainly from slower marginal calibration.

## Controls

- same tokenizer-fit raw bytes,
- same measured stream,
- same 4082-class model width,
- same model architecture and seed,
- same shared BPE/Unigram raw-byte optimizer boundaries,
- same `1e-3` AdamW learning rate,
- same smoothing rule for both tokenizers,
- no measured-stream information used to construct the prior.

## Status

Implementation pending.
