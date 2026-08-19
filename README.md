# BPE vs Tunstall-style tokens: block-prequential WikiText experiment

This is intentionally **not a framework**. It asks one question:

> With the same flat-softmax decoder-only Transformer and approximately the same vocabulary size, does a prefix-free Tunstall-style byte phrase vocabulary produce a better block-prequential codelength than byte-level BPE on WikiText?

The primary metric is **prequential bits per raw UTF-8 byte**. Token perplexity is not used for the comparison.

## What is held fixed

- Same raw WikiText byte ranges at every prequential stage.
- Same model vocabulary size for both tokenizers.
- Same Transformer architecture, initialization seed, token context, optimizer, and batching.
- Exactly **one training pass over each observed prefix**. There is no epoch parameter.
- Same tokenizer-training prefix. The tokenizer is treated as known side information and its own description length is not included.

## Tunstall vocabulary

The Tunstall phrase tree contains **bytes only**. It starts with 256 one-byte leaves. Expanding one leaf replaces it with all 256 possible one-byte continuations, so each expansion adds 255 phrase leaves.

EOS is a **separate flat-softmax token** and is not a branch in the Tunstall tree. Therefore legal model vocabulary sizes are

```text
V = 1 + (256 + 255*k) = 257 + 255*k
```

A requested size is snapped to the nearest legal size. For example, `4096` becomes `4082`.

There are three construction modes:

- `--tunstall-mode boundary` (default): tokenize the fit corpus with the current prefix-free tree, count leaves actually emitted at tokenizer boundaries, expand the most frequent one, then retokenize and repeat.
- `--tunstall-mode empirical`: use phrase-prefix frequency at arbitrary byte positions.
- `--tunstall-mode iid`: classical Tunstall using the empirical IID byte distribution.

The first mode directly chases the experiment's motivating idea: keep splitting the currently overrepresented token region.

### Finite-message boundaries

A byte-only prefix-free phrase tree can end a finite byte string part-way through an internal phrase. We do **not** add EOS branches or fallback prefix tokens to hide that.

Instead, requested prequential cuts are moved backward by a few bytes to the nearest position that is both:

1. a completed Tunstall phrase, and
2. a UTF-8 boundary.

Those exact raw-byte cuts are then used for **both** Tunstall and BPE. EOS remains one separate model token rather than consuming 256 extra branches throughout the phrase tree.

## Prequential protocol

Default block endpoints are logarithmically spaced:

```text
0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0
```

For each tokenizer:

1. Encode the first block under a uniform token code: `N_tokens * log2(V)` bits.
2. For each later block, initialize the same Transformer from the same seed.
3. Train it from scratch for **one pass only** over all bytes revealed before that block.
4. Score the next byte-identical block by causal token NLL.
5. Sum block codelengths and divide by the exact number of raw UTF-8 bytes encoded.

Each evaluation block begins without cross-block model context, so its first token uses a uniform code. This is a block-prequential NLL experiment, not an implemented arithmetic coder.

## Run

```bash
uv sync
uv run python experiment.py
```

Defaults:

- WikiText-2 raw
- requested vocab ~4096 (`4082` exactly after snapping)
- 2 MB tokenizer-fit prefix
- **all remaining WikiText-2 train bytes** for prequential evaluation
- 4-layer, 256-wide Transformer
- context 256 tokens
- one training pass per observed prefix

A smaller smoke test:

```bash
uv run python experiment.py \
  --max-preq-mb 0.5 \
  --tokenizer-fit-mb 0.5 \
  --d-model 128 \
  --layers 2 \
  --heads 4 \
  --context 128 \
  --batch-size 8
```

A larger WikiText-103 run:

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
  --batch-size 16
```

## Output

The console prints tokenizer diagnostics:

- bytes/token
- `H(T) / log2(V)` — marginal token entropy relative to uniform
- number of vocabulary entries observed

Then it prints every prequential block and its cumulative code.

`results.json` contains the full stage records plus a compact `code_curve` for each tokenizer. Each point stores:

- raw bytes encoded
- cumulative prequential bits
- cumulative bits/byte
- bytes/tokens used to train that stage's model
- optimizer steps for that one-pass model fit
- cumulative optimizer steps across the whole experiment
- cumulative train+score wall time

That lets us compare the prequential code later against data seen, model steps, or experiment compute without rerunning.
