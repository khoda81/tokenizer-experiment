# Experiment 002 — Sparse prefix refinement ("Bunstall")

Date: 2026-08-19

## Hypothesis

Experiment 001 showed that a complete byte-level Tunstall expansion is extremely expensive under a ~4k vocabulary budget: every selected prefix must allocate all 256 next-byte children, consuming 255 additional phrase tokens even when only a handful of continuations carry meaningful probability.

BPE appeared to behave more like a sparse decision tree. A merge `A + B -> AB` spends one vocabulary slot on the event "after A, continue with B" while occurrences of A with other continuations remain represented without allocating sibling phrases.

This experiment isolates that structural idea.

## Prototype

`SparsePrefixTokenizer` (nickname **Bunstall**) starts with all 256 byte tokens plus a separate EOS token. It then promotes one continuation at a time:

```text
p  ->  p remains a token
       pb becomes one additional token
```

Parsing is deterministic greedy longest-match over the resulting prefix trie. For example, retaining `and` while adding `andr` implements the decision "continue past `and` when the next byte is `r`; otherwise stop at `and`" without allocating the other 255 byte continuations.

This means the vocabulary is deliberately **not prefix-free**. The prototype therefore does not retain Tunstall's direct mapping from a flat token distribution to a left-to-right byte distribution. The purpose of this experiment is narrower: test whether sparse prefix refinement explains BPE's vocabulary efficiency.

## Training heuristics

Two cheap arbitrary-position heuristics are available:

- `entropy`: choose the continuation with maximum local residual binary-partition entropy gain

  ```text
  residual_count(p) * h2(count(pb) / residual_count(p))
  ```

- `frequency`: choose the most frequent candidate continuation, closer in spirit to BPE's frequency-driven merges.

Counts are measured at arbitrary corpus positions rather than exact emitted-token boundaries. This is intentional for the first structural prototype; if the result is promising, the training objective can be made parsing-aware.

## Run

```bash
uv run python scripts/inspect_bunstall.py
```

Compare the frequency heuristic with:

```bash
uv run python scripts/inspect_bunstall.py --mode frequency
```

The runner reports:

- bytes/token,
- `H(T) / log2(V)`,
- unigram bits/raw-byte,
- vocabulary utilization,
- representative emitted phrases,
- the distribution of binary split probabilities and entropies,
- highest-support and most-balanced promoted continuations.

The first question is simply whether Bunstall closes the large marginal-efficiency gap from Experiment 001 before spending GPU time on another prequential Transformer run.
