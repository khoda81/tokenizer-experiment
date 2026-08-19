# BPE vs Tunstall-style tokens: block-prequential WikiText experiment

This is intentionally **not a framework**. It is one small experiment answering:

> With the same flat-softmax decoder-only Transformer and the same vocabulary size, does a prefix-free Tunstall-style phrase vocabulary produce a better block-prequential codelength than byte-level BPE on WikiText?

The primary metric is **prequential bits per raw UTF-8 byte**. Token perplexity is not used for the comparison.

## What is held fixed

- Same raw WikiText byte ranges at every prequential stage.
- Same vocabulary size. A Tunstall tree over 256 bytes plus EOS can only have `257 + 256*k` leaves, so `--vocab-size` is snapped to the nearest legal value and BPE is trained to exactly that size. `4096` becomes `4097`.
- Same Transformer architecture, initialization seed, context length in tokens, optimizer, epochs, and batching.
- Same tokenizer-training prefix. The tokenizer is treated as known side information and **its own description length is not included**.

The two tokenizers differ only in the segmentation/vocabulary they expose to the flat softmax.

## Tunstall implementation

There are two modes:

- `--tunstall-mode boundary` (default): tokenize the fit corpus with the current prefix-free tree, count the leaves that are **actually emitted at tokenizer boundaries**, expand the most frequent one into all 257 continuations, then retokenize and repeat. This is the literal experiment we discussed: keep splitting the token that most hurts context-free marginal uniformity.
- `--tunstall-mode empirical`: faster phrase-prefix-frequency variant; expand the leaf with the highest occurrence count at arbitrary byte positions. Useful for larger vocabulary/tokenizer-fit sweeps.
- `--tunstall-mode iid`: classical Tunstall construction using the empirical IID byte distribution, i.e. phrase probability is the product of byte unigram probabilities.

The `boundary` and `empirical` modes are deliberately called **Tunstall-style**, not claimed to be optimal generalized Tunstall codes for a source with memory.

The resulting leaf dictionary is prefix-free. EOS is never expanded. Appending EOS means every finite text block terminates at a leaf, so the tokenizer is lossless and deterministic.

One tiny asymmetry to know about: BPE's EOS is a dedicated special token, while a Tunstall leaf may include the final bytes of a block together with EOS. There are only a handful of prequential block boundaries, so this is negligible for MB-scale blocks, but it is recorded here rather than hidden.

## Prequential protocol

The default byte fractions are:

```text
0.05, 0.15, 0.35, 0.65, 1.0
```

For each tokenizer:

1. Encode the first 5% under a uniform code: `N_tokens * log2(V)` bits.
2. For each later block, initialize the same Transformer from the same seed.
3. Train it **from scratch** on all bytes revealed before that block (tokenized with the candidate tokenizer).
4. Score the next byte-identical block by causal token NLL.
5. Sum the block codelengths and divide by the exact number of raw UTF-8 bytes encoded.

Each evaluation block is treated as a fresh message. Its first token is sent with a uniform prior; subsequent tokens are predicted causally within the block. This avoids giving either tokenizer awkward cross-block partial-token state.

This is a block-prequential NLL experiment, not an implemented arithmetic coder. Reproducing the exact floating-point GPU training run bit-for-bit at a decoder is outside scope.

## Run it

With `uv`:

```bash
uv sync
uv run python experiment.py
```

The defaults are intentionally modest:

- WikiText-2 raw
- ~4097 vocabulary
- 2 MB tokenizer-fit prefix
- 4 MB prequential stream
- 4-layer, 256-wide Transformer
- context 256
- one training epoch per prequential stage

That should be a useful first GPU run rather than a week-long commitment.

### Very fast smoke test

```bash
uv run python experiment.py \
  --max-preq-mb 0.5 \
  --tokenizer-fit-mb 0.5 \
  --d-model 128 \
  --layers 2 \
  --heads 4 \
  --context 128 \
  --batch-size 8 \
  --max-train-steps 20
```

### A more serious WikiText-103 run

For a larger GPU, e.g. a 4090:

```bash
uv run python experiment.py \
  --dataset-config wikitext-103-raw-v1 \
  --vocab-size 16384 \
  --tokenizer-fit-mb 32 \
  --max-preq-mb 64 \
  --d-model 512 \
  --layers 8 \
  --heads 8 \
  --context 512 \
  --batch-size 16 \
  --epochs 1
```

`16384` snaps to `16385 = 257 + 63*256`.

If memory is tight, reduce `--batch-size` first. If you want a quick trend before committing to full epochs, use `--max-train-steps`.

## Output

The console prints tokenizer diagnostics first:

- bytes/token
- `H(T) / log2(V)` — how close the marginal token distribution is to uniform
- number of vocabulary entries actually observed

Then it prints the bits/byte of every prequential block and finally:

```text
FINAL
  bpe                  ... bits/byte
  tunstall-boundary    ... bits/byte
  Tunstall - BPE       ... bits/byte
```

A machine-readable `results.json` is written by default.

## Useful follow-ups if the result is interesting

Do **not** abstract the API yet. First vary only one thing at a time:

1. Repeat 3–5 seeds.
2. Compare boundary-frequency vs prefix-frequency vs classical-IID Tunstall.
3. Sweep vocab sizes (`4097`, `8193`, `16385`, ...).
4. Plot learning curves by prequential block.
5. Compare fixed epochs vs fixed optimizer-step/FLOP budgets.
6. Only then worry about making BPE→byte exact and plugging both into a common byte-level PoE.
