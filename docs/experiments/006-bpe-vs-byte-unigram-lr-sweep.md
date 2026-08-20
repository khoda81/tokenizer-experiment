# Experiment 006 — Byte-Unigram vs BPE learning-rate sweep

Date: 2026-08-21  
Status: **complete**

## Motivation

Experiment 005 found that a byte-level Unigram LM tokenizer beat byte-level BPE on both empirical unigram codelength and cumulative continuous-stream prequential codelength. Byte-Unigram reached 2.2579 bpb while the matched BPE run reached 2.3118 bpb.

However, interval analysis showed that the gap was much larger early in the stream and shrank toward the tail. This suggested that much of the cumulative advantage might come from faster online adaptation: the Unigram tokenizer has already encoded more zero-order structure, while BPE asks gradient descent to learn more of it.

## Question

Does Byte-Unigram retain its cumulative and late-stream advantage when the Transformer learning rate changes?

A simple learning-rate explanation predicts that BPE should close most of the gap once its optimizer is sufficiently aggressive. A persistent late-stream gap near the best common learning rate would be stronger evidence that Byte-Unigram presents an easier finite-model conditional prediction problem, not merely a better starting marginal distribution.

## Protocol

Only two tokenizers are active:

- byte-level Unigram LM,
- byte-level BPE.

The tokenizer-fit prefix, measured continuous byte stream, model architecture, seed, weight decay, context, and raw-byte update schedule are shared. Both tokenizers are trained once. Each learning rate receives a freshly initialized Transformer with the same seed.

Sweep:

```text
3e-4, 1e-3, 3e-3
```

Model vocabulary width remains 4082: 4081 source-token slots plus one BOS-only model class.

With only BPE and Byte-Unigram participating in the boundary intersection, the optimizer schedule is almost exactly the intended 256 raw bytes/update:

```text
34,970 updates
mean 256.0 raw bytes/update
min 130
max 284
```

The experiment reports cumulative prequential bits/raw-byte and exact tail rates over approximately 250 KB, 500 KB, 1 MB, 2 MB, and 4 MB.

## Results

### Cumulative prequential codelength

| learning rate | Byte-Unigram | BPE | Unigram - BPE | relative reduction vs BPE |
| ---: | ---: | ---: | ---: | ---: |
| 3e-4 | **2.286530** | 2.334183 | -0.047652 | 2.04% |
| 1e-3 | **2.251977** | 2.318919 | -0.066943 | **2.89%** |
| 3e-3 | **2.286515** | 2.368855 | -0.082341 | 3.48% |

The best absolute learning rate for **both** tokenizers is 1e-3. Therefore Byte-Unigram's advantage is not explained by BPE simply needing a larger learning rate than the original experiment used.

### Late-stream rates

Sign convention below is `Unigram - BPE`, so negative values favor Byte-Unigram.

| learning rate | last ~1 MB | last ~500 KB | last ~250 KB |
| ---: | ---: | ---: | ---: |
| 3e-4 | +0.004287 | +0.013407 | +0.017947 |
| 1e-3 | **-0.026123** | **-0.019185** | **-0.016235** |
| 3e-3 | **-0.053968** | **-0.044056** | **-0.038120** |

At 3e-4, BPE eventually catches and slightly overtakes Byte-Unigram locally even though Byte-Unigram still wins cumulative code. At the shared optimum 1e-3, Byte-Unigram retains a clear late-stream advantage. At 3e-3 both tokenizers degrade in absolute performance, but BPE degrades substantially more and the Byte-Unigram gap grows.

## Interpretation

The sweep rejects the simplest version of the learning-rate confound:

> BPE does not lose merely because the original 1e-3 learning rate was too small for it.

The same 1e-3 rate minimizes cumulative codelength for both tokenizers, and Byte-Unigram still wins both cumulative and tail performance there.

The results do show that optimization dynamics matter substantially:

- at low LR, the early Unigram advantage decays through zero locally;
- near the shared optimum, a smaller but persistent local advantage remains;
- at excessive LR, BPE is less robust and the gap grows.

This is consistent with Byte-Unigram presenting an easier online optimization problem, but the experiment still conflates two effects:

1. learning each tokenizer's zero-order marginal distribution;
2. learning contextual corrections beyond that marginal.

The next experiment should remove the first factor explicitly by giving both models their tokenizer-specific unigram prior from the tokenizer-fit side information.

## Run

```bash
HF_HUB_OFFLINE=1 uv run python scripts/run_experiment.py \
  --lrs 3e-4,1e-3,3e-3 \
  --wandb-mode offline \
  --wandb-run-name bpe-unigram-lr-sweep
```

## Outcome

Byte-Unigram is robustly better in cumulative prequential codelength across the entire LR sweep. The best shared operating point is 1e-3, where Byte-Unigram improves BPE by about **0.0669 bpb / 2.89%** and remains better in the final 250 KB–1 MB of the stream.

This motivates Experiment 007: initialize each model with a unigram logit prior estimated only from side information and measure whether the residual Byte-Unigram advantage survives.
