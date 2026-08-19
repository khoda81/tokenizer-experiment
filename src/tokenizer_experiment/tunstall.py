from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

BYTE_ALPHABET = 256
EOS_TOKEN = 256  # conceptual symbol only; EOS is not a Tunstall-tree branch


@dataclass
class _Node:
    phrase: tuple[int, ...]
    count: int
    children: list[int] | None = None
    token_id: int | None = None
    positions: np.ndarray | None = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return self.children is None


class EmpiricalTunstallTokenizer:
    """A prefix-free phrase tokenizer whose tree contains bytes only.

    The phrase tree begins with one leaf per byte. Repeatedly, the selected leaf
    is replaced by all 256 one-byte continuations, so each expansion adds 255
    phrase tokens. EOS is one separate model token and is not a tree branch.

    Model vocabulary size: V = 1 + (256 + 255*k) = 257 + 255*k.
    """

    def __init__(self, nodes: list[_Node], root: int, phrase_vocab_size: int, mode: str):
        self.nodes = nodes
        self.root = root
        self.phrase_vocab_size = phrase_vocab_size
        self.eos_id = phrase_vocab_size
        self.vocab_size = phrase_vocab_size + 1
        self.mode = mode
        self._id_to_phrase: list[tuple[int, ...]] = [()] * phrase_vocab_size
        for node in nodes:
            if node.token_id is not None:
                self._id_to_phrase[node.token_id] = node.phrase

    @staticmethod
    def legal_vocab_size(requested: int) -> int:
        if requested < BYTE_ALPHABET + 1:
            raise ValueError(f"vocab_size must be >= {BYTE_ALPHABET + 1}")
        k = max(0, round((requested - (BYTE_ALPHABET + 1)) / (BYTE_ALPHABET - 1)))
        return BYTE_ALPHABET + 1 + k * (BYTE_ALPHABET - 1)

    @classmethod
    def train(
        cls,
        data: bytes,
        requested_vocab_size: int,
        mode: Literal["boundary", "empirical", "iid"] = "boundary",
    ) -> EmpiricalTunstallTokenizer:
        model_vocab_size = cls.legal_vocab_size(requested_vocab_size)
        phrase_target = model_vocab_size - 1
        symbols = np.frombuffer(data, dtype=np.uint8)
        counts = np.bincount(symbols.astype(np.int64), minlength=BYTE_ALPHABET)

        nodes: list[_Node] = [_Node(phrase=(), count=len(symbols), children=[])]
        root = 0
        root_children: list[int] = []
        heap: list[tuple[float, int, int]] = []
        serial = 0

        probs = counts.astype(np.float64) / max(1, counts.sum()) if mode == "iid" else None

        for sym in range(BYTE_ALPHABET):
            node_id = len(nodes)
            node = _Node(phrase=(sym,), count=int(counts[sym]))
            nodes.append(node)
            root_children.append(node_id)
            if mode != "boundary" and node.count > 0:
                priority = cls._priority(node, mode, probs)
                heapq.heappush(heap, (-priority, serial, node_id))
                serial += 1
        nodes[root].children = root_children

        leaf_count = BYTE_ALPHABET
        expansions = (phrase_target - BYTE_ALPHABET) // (BYTE_ALPHABET - 1)

        for _ in range(expansions):
            if mode == "boundary":
                emitted = cls._leaf_counts(nodes, root, data)
                candidates = [
                    (count, leaf_id)
                    for leaf_id, count in emitted.items()
                    if count > 0 and nodes[leaf_id].is_leaf
                ]
                if not candidates:
                    raise RuntimeError("No expandable Tunstall leaves remain")
                _, node_id = max(candidates)
            else:
                if not heap:
                    raise RuntimeError("No expandable Tunstall leaves remain")
                _, _, node_id = heapq.heappop(heap)

            node = nodes[node_id]
            if not node.is_leaf:
                raise AssertionError("selected a non-leaf Tunstall node")

            child_ids: list[int] = []
            if mode == "empirical":
                positions = node.positions
                if positions is None:
                    first = node.phrase[0]
                    positions = np.flatnonzero(symbols == first).astype(np.int32)
                    if len(node.phrase) > 1:
                        mask = np.ones(len(positions), dtype=bool)
                        for depth, sym in enumerate(node.phrase[1:], start=1):
                            idx = positions.astype(np.int64) + depth
                            in_range = idx < len(symbols)
                            matches = np.zeros(len(positions), dtype=bool)
                            valid_rows = np.flatnonzero(in_range)
                            matches[valid_rows] = symbols[idx[in_range]] == sym
                            mask &= matches
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
                        int(sym): sorted_pos[start : start + count].astype(np.int32, copy=True)
                        for sym, start, count in zip(unique, starts, child_counts)
                    }
                else:
                    groups = {}
            else:
                groups = None

            for sym in range(BYTE_ALPHABET):
                phrase = node.phrase + (sym,)
                if mode == "empirical":
                    pos = groups.get(sym)
                    count = 0 if pos is None else len(pos)
                else:
                    count = 0
                    pos = None

                child_id = len(nodes)
                child = _Node(phrase=phrase, count=count, positions=pos)
                nodes.append(child)
                child_ids.append(child_id)

                if mode != "boundary":
                    priority = cls._priority(child, mode, probs)
                    if priority > 0:
                        heapq.heappush(heap, (-priority, serial, child_id))
                        serial += 1

            node.children = child_ids
            node.positions = None
            leaf_count += BYTE_ALPHABET - 1

        if leaf_count != phrase_target:
            raise AssertionError((leaf_count, phrase_target))

        token_id = 0
        for node in nodes:
            if node.is_leaf:
                node.token_id = token_id
                token_id += 1
        if token_id != phrase_target:
            raise AssertionError((token_id, phrase_target))

        return cls(nodes, root, phrase_target, mode)

    @staticmethod
    def _leaf_counts(nodes: list[_Node], root: int, data: bytes) -> dict[int, int]:
        counts: dict[int, int] = {}
        node_id = root
        for sym in data:
            children = nodes[node_id].children
            if children is None:
                raise AssertionError("expected internal node before consuming byte")
            node_id = children[sym]
            if nodes[node_id].is_leaf:
                counts[node_id] = counts.get(node_id, 0) + 1
                node_id = root
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

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"), add_eos=add_eos)

    def encode_bytes(self, data: bytes, *, add_eos: bool = False) -> list[int]:
        out: list[int] = []
        node_id = self.root
        for sym in data:
            children = self.nodes[node_id].children
            if children is None:
                raise AssertionError("expected internal node before consuming byte")
            node_id = children[sym]
            node = self.nodes[node_id]
            if node.is_leaf:
                assert node.token_id is not None
                out.append(node.token_id)
                node_id = self.root
        if node_id != self.root:
            raise ValueError("byte string ends inside a Tunstall phrase; align the boundary first")
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        out = bytearray()
        saw_eos = False
        for token_id in ids:
            if token_id == self.eos_id:
                if saw_eos:
                    raise ValueError("multiple EOS tokens")
                saw_eos = True
                continue
            if saw_eos:
                raise ValueError("token after EOS")
            out.extend(self._id_to_phrase[token_id])
        return bytes(out)

    def align_utf8_boundaries(self, data: bytes, fractions: list[float]) -> list[int]:
        """Move requested cuts backward to completed phrase + UTF-8 boundaries."""
        targets = [len(data) if f >= 1.0 else round(len(data) * f) for f in fractions]
        out: list[int] = []
        target_i = 0
        last_good = 0
        node_id = self.root

        for pos, sym in enumerate(data, start=1):
            while target_i < len(targets) and targets[target_i] < pos:
                if last_good <= (out[-1] if out else 0):
                    raise ValueError("aligned prequential boundary collapsed")
                out.append(last_good)
                target_i += 1

            children = self.nodes[node_id].children
            if children is None:
                raise AssertionError("expected internal node before consuming byte")
            node_id = children[sym]
            if self.nodes[node_id].is_leaf:
                node_id = self.root
                if pos == len(data) or data[pos] & 0xC0 != 0x80:
                    last_good = pos

            while target_i < len(targets) and targets[target_i] == pos:
                if last_good <= (out[-1] if out else 0):
                    raise ValueError("aligned prequential boundary collapsed")
                out.append(last_good)
                target_i += 1

        while target_i < len(targets):
            if last_good <= (out[-1] if out else 0):
                raise ValueError("aligned prequential boundary collapsed")
            out.append(last_good)
            target_i += 1

        return out

    def token_piece(self, token_id: int) -> tuple[int, ...]:
        return () if token_id == self.eos_id else self._id_to_phrase[token_id]

    def max_phrase_bytes(self) -> int:
        return max(len(p) for p in self._id_to_phrase)

    def assert_prefix_free(self) -> None:
        phrases = sorted(self._id_to_phrase)
        for a, b in itertools.pairwise(phrases):
            if len(a) <= len(b) and b[: len(a)] == a:
                raise AssertionError(f"not prefix-free: {a} prefixes {b}")


class BPETokenizer:
    """Thin byte-level BPE wrapper with the same model vocabulary size."""

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
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=[cls.EOS_TOKEN],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        char_chunk = 8_192
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

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if add_eos:
            ids.append(self.eos_id)
        return ids
