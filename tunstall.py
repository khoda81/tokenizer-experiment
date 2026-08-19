from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import chain
from typing import Literal

import numpy as np

EOS = 256
ALPHABET = 257  # 256 bytes + an end-of-message symbol


@dataclass
class _Node:
    phrase: tuple[int, ...]
    count: int
    children: list[int] | None = None
    token_id: int | None = None
    # Occurrence starts in the tokenizer-fit stream. Kept only when cheaply
    # inherited from an expanded parent; root children can lazily recover them.
    positions: np.ndarray | None = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def contains_eos(self) -> bool:
        return EOS in self.phrase


class EmpiricalTunstallTokenizer:
    """A complete prefix-free phrase tokenizer over bytes.

    The tree begins with one leaf per byte plus EOS. Repeatedly, the most
    probable current leaf is replaced by all 257 one-symbol continuations.

    `mode="iid"` is classical Tunstall under an empirical IID byte source.
    `mode="boundary"` is the default experiment: retokenize the fit corpus,
    expand the most frequent *actually emitted* leaf, and repeat. It directly
    chases a flatter context-free token marginal.
    `mode="empirical"` is a faster phrase-prefix-frequency variant using
    occurrences at arbitrary byte positions. Neither non-IID mode is claimed
    to be an optimal generalized Tunstall code.

    EOS is never expanded. Therefore every finite byte string can be encoded:
    append EOS, walk the tree until a leaf, emit it, and restart at the root.
    """

    def __init__(self, nodes: list[_Node], root: int, vocab_size: int, mode: str):
        self.nodes = nodes
        self.root = root
        self.vocab_size = vocab_size
        self.mode = mode
        self._id_to_phrase: list[tuple[int, ...]] = [()] * vocab_size
        for node in nodes:
            if node.token_id is not None:
                self._id_to_phrase[node.token_id] = node.phrase

    @staticmethod
    def legal_vocab_size(requested: int) -> int:
        if requested < ALPHABET:
            raise ValueError(f"vocab_size must be >= {ALPHABET}")
        k = max(0, round((requested - ALPHABET) / (ALPHABET - 1)))
        return ALPHABET + k * (ALPHABET - 1)

    @classmethod
    def train(
        cls,
        data: bytes,
        requested_vocab_size: int,
        mode: Literal["boundary", "empirical", "iid"] = "boundary",
    ) -> EmpiricalTunstallTokenizer:
        target = cls.legal_vocab_size(requested_vocab_size)
        symbols = np.empty(len(data) + 1, dtype=np.uint16)
        symbols[:-1] = np.frombuffer(data, dtype=np.uint8)
        symbols[-1] = EOS

        counts = np.bincount(symbols.astype(np.int64), minlength=ALPHABET)
        nodes: list[_Node] = [_Node(phrase=(), count=len(symbols), children=[])]
        root = 0
        root_children: list[int] = []
        heap: list[tuple[float, int, int]] = []
        serial = 0

        if mode == "iid":
            probs = counts.astype(np.float64) / counts.sum()
        else:
            probs = None

        for sym in range(ALPHABET):
            node_id = len(nodes)
            node = _Node(phrase=(sym,), count=int(counts[sym]))
            nodes.append(node)
            root_children.append(node_id)
            if mode != "boundary" and sym != EOS and node.count > 0:
                priority = cls._priority(node, mode, probs)
                heapq.heappush(heap, (-priority, serial, node_id))
                serial += 1
        nodes[root].children = root_children

        leaf_count = ALPHABET
        expansions = (target - ALPHABET) // (ALPHABET - 1)

        for _ in range(expansions):
            if mode == "boundary":
                # Literal objective for this experiment: look at the tokens the
                # current tree actually emits, then split the most frequent one.
                # Retokenizing after each split is intentionally dumb but exact.
                emitted = cls._leaf_counts(nodes, root, data)
                candidates = [
                    (count, leaf_id)
                    for leaf_id, count in emitted.items()
                    if count > 0
                    and nodes[leaf_id].is_leaf
                    and not nodes[leaf_id].contains_eos
                ]
                if not candidates:
                    raise RuntimeError("No expandable Tunstall leaves remain")
                _, node_id = max(candidates)
            else:
                if not heap:
                    raise RuntimeError("No expandable Tunstall leaves remain")
                _, _, node_id = heapq.heappop(heap)

            node = nodes[node_id]
            if not node.is_leaf or node.contains_eos:
                raise AssertionError("selected a non-expandable node")

            child_ids: list[int] = []
            if mode == "empirical":
                positions = node.positions
                if positions is None:
                    # Only root children should normally reach this path.
                    first = node.phrase[0]
                    positions = np.flatnonzero(symbols[:-1] == first).astype(np.int32)
                    # If this is unexpectedly deeper, refine by the remainder.
                    if len(node.phrase) > 1:
                        mask = np.ones(len(positions), dtype=bool)
                        for depth, sym in enumerate(node.phrase[1:], start=1):
                            idx = positions.astype(np.int64) + depth
                            mask &= idx < len(symbols)
                            valid_idx = idx[mask]
                            tmp = np.zeros_like(mask)
                            tmp[np.flatnonzero(mask)] = symbols[valid_idx] == sym
                            mask = tmp
                        positions = positions[mask]

                next_idx = positions.astype(np.int64) + len(node.phrase)
                valid = next_idx < len(symbols)
                parent_positions = positions[valid]
                next_syms = symbols[next_idx[valid]]

                if len(parent_positions):
                    order = np.argsort(next_syms, kind="stable")
                    sorted_syms = next_syms[order]
                    sorted_pos = parent_positions[order]
                    unique, starts, child_counts = np.unique(
                        sorted_syms, return_index=True, return_counts=True
                    )
                    groups = {
                        int(sym): sorted_pos[start : start + count].astype(
                            np.int32, copy=True
                        )
                        for sym, start, count in zip(unique, starts, child_counts)
                    }
                else:
                    groups = {}
            else:
                groups = None

            for sym in range(ALPHABET):
                phrase = node.phrase + (sym,)
                if mode == "empirical":
                    pos = groups.get(sym)
                    count = 0 if pos is None else len(pos)
                else:
                    # Keep count only for diagnostics. The heap uses IID mass.
                    count = 0
                    pos = None

                child_id = len(nodes)
                child = _Node(phrase=phrase, count=count, positions=pos)
                nodes.append(child)
                child_ids.append(child_id)

                if mode != "boundary" and sym != EOS:
                    priority = cls._priority(child, mode, probs)
                    if priority > 0:
                        heapq.heappush(heap, (-priority, serial, child_id))
                        serial += 1

            node.children = child_ids
            node.positions = None
            leaf_count += ALPHABET - 1

        if leaf_count != target:
            raise AssertionError((leaf_count, target))

        token_id = 0
        for node in nodes:
            if node.is_leaf:
                node.token_id = token_id
                token_id += 1
        if token_id != target:
            raise AssertionError((token_id, target))

        return cls(nodes, root, target, mode)

    @staticmethod
    def _leaf_counts(nodes: list[_Node], root: int, data: bytes) -> dict[int, int]:
        counts: dict[int, int] = {}
        node_id = root
        for sym in chain(data, (EOS,)):
            children = nodes[node_id].children
            if children is None:
                raise AssertionError("expected internal node before consuming symbol")
            node_id = children[sym]
            if nodes[node_id].is_leaf:
                counts[node_id] = counts.get(node_id, 0) + 1
                node_id = root
        if node_id != root:
            raise AssertionError("EOS failed to terminate tokenizer-fit stream")
        return counts

    @staticmethod
    def _priority(node: _Node, mode: str, probs: np.ndarray | None) -> float:
        if mode == "empirical":
            return float(node.count)
        assert probs is not None
        p = 1.0
        for sym in node.phrase:
            p *= float(probs[sym])
        return p

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        return self.encode_bytes(data)

    def encode_bytes(self, data: bytes) -> list[int]:
        # EOS makes the phrase dictionary complete for finite messages.
        symbols: Iterable[int] = chain(data, (EOS,))
        out: list[int] = []
        node_id = self.root
        for sym in symbols:
            node = self.nodes[node_id]
            if node.children is None:
                raise AssertionError("expected internal node before consuming a symbol")
            node_id = node.children[sym]
            node = self.nodes[node_id]
            if node.is_leaf:
                assert node.token_id is not None
                out.append(node.token_id)
                node_id = self.root
        if node_id != self.root:
            raise AssertionError("EOS failed to terminate final phrase")
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        symbols: list[int] = []
        for token_id in ids:
            symbols.extend(self._id_to_phrase[token_id])
        if not symbols or symbols[-1] != EOS:
            raise ValueError("encoded message does not terminate in EOS")
        if EOS in symbols[:-1]:
            raise ValueError("EOS appeared before end of message")
        return bytes(symbols[:-1])

    def token_piece(self, token_id: int) -> tuple[int, ...]:
        return self._id_to_phrase[token_id]

    def max_phrase_bytes(self) -> int:
        return max(sum(sym != EOS for sym in p) for p in self._id_to_phrase)

    def assert_prefix_free(self) -> None:
        phrases = sorted(self._id_to_phrase)
        for a, b in zip(phrases, phrases[1:]):
            if len(a) <= len(b) and b[: len(a)] == a:
                raise AssertionError(f"not prefix-free: {a} prefixes {b}")


class BPETokenizer:
    """Thin byte-level BPE wrapper with the same vocabulary size."""

    EOS_TOKEN = "<|eos|>"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        eos = tokenizer.token_to_id(self.EOS_TOKEN)
        if eos is None:
            raise RuntimeError("BPE tokenizer has no EOS token")
        self.eos_id = eos

    @classmethod
    def train(cls, text: str, vocab_size: int) -> BPETokenizer:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.BPE(unk_token=None))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=False,
        )
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=[cls.EOS_TOKEN],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # Chunk only to keep tokenizer training memory tame. use_regex=False
        # avoids GPT-2 word splitting; the only lost merges are across chunk edges.
        char_chunk = 1_000_000
        iterator = (text[i : i + char_chunk] for i in range(0, len(text), char_chunk))
        tokenizer.train_from_iterator(
            iterator, trainer=trainer, length=math.ceil(len(text) / char_chunk)
        )
        wrapped = cls(tokenizer)
        if wrapped.vocab_size != vocab_size:
            raise RuntimeError(
                f"BPE trainer produced vocab={wrapped.vocab_size}, expected {vocab_size}. "
                "Try a larger tokenizer-fit prefix or a smaller vocabulary."
            )
        return wrapped

    def encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        ids.append(self.eos_id)
        return ids
