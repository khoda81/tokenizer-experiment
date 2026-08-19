from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class _Node:
    phrase: bytes
    token_id: int
    children: dict[int, int] = field(default_factory=dict)
    continuation_counts: np.ndarray | None = field(default=None, repr=False)
    version: int = 0


@dataclass(frozen=True)
class Expansion:
    parent: bytes
    child: bytes
    byte: int
    child_occurrences: int
    residual_occurrences: int
    q: float
    binary_entropy_bits: float
    score: float


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


class SparsePrefixTokenizer:
    """Maximal-munch byte tokenizer with selectively promoted prefix children.

    Nickname: "Bunstall".

    Unlike Tunstall, promoting ``p + b`` does *not* replace ``p`` or allocate
    all 256 continuations. ``p`` remains a valid token and one additional
    vocabulary slot is spent on exactly one continuation. Parsing is greedy
    longest-match, so at prefix ``p`` the decision is effectively "continue
    with promoted byte b (or another promoted child) vs stop at p".

    This vocabulary is intentionally *not prefix-free*. It is a controlled
    experiment for testing whether sparse prefix refinement explains BPE's
    vocabulary efficiency; it does not retain Tunstall's direct flat-token to
    next-byte probability mapping.

    Training uses arbitrary-position corpus continuation counts as a cheap
    structural heuristic. ``mode="entropy"`` greedily maximizes the local
    binary partition entropy gained per one vocabulary slot. ``mode="frequency"``
    greedily promotes the most frequent continuation instead.
    """

    def __init__(
        self,
        nodes: list[_Node],
        root_children: dict[int, int],
        vocab_size: int,
        expansions: list[Expansion],
        mode: str,
    ) -> None:
        self.nodes = nodes
        self.root_children = root_children
        self.vocab_size = vocab_size
        self.eos_id = vocab_size - 1
        self.expansions = expansions
        self.mode = mode
        self._id_to_phrase = [b""] * (vocab_size - 1)
        for node in nodes:
            self._id_to_phrase[node.token_id] = node.phrase

    @classmethod
    def train(
        cls,
        data: bytes,
        vocab_size: int,
        *,
        mode: Literal["entropy", "frequency"] = "entropy",
    ) -> SparsePrefixTokenizer:
        if vocab_size < 257:
            raise ValueError("vocab_size must leave room for 256 bytes plus EOS")
        if not data:
            raise ValueError("cannot train on empty data")

        nodes: list[_Node] = []
        root_children: dict[int, int] = {}
        for byte in range(256):
            node_id = len(nodes)
            nodes.append(_Node(phrase=bytes([byte]), token_id=byte))
            root_children[byte] = node_id

        # Heap entries are (-score, -support, serial, parent_id, byte, version).
        # `version` makes sibling scores lazily invalidated whenever one child
        # consumes part of the parent's residual "not yet promoted" mass.
        heap: list[tuple[float, int, int, int, int, int]] = []
        serial = 0

        def continuation_counts(phrase: bytes) -> np.ndarray:
            counts = np.zeros(256, dtype=np.int64)
            start = 0
            phrase_len = len(phrase)
            while True:
                pos = data.find(phrase, start)
                if pos < 0:
                    break
                next_pos = pos + phrase_len
                if next_pos < len(data):
                    counts[data[next_pos]] += 1
                # Allow overlapping occurrences; phrases are short and this is
                # only tokenizer-fit analysis, so the simple implementation is
                # preferable to a suffix-index dependency.
                start = pos + 1
            return counts

        def ensure_counts(node_id: int) -> np.ndarray:
            node = nodes[node_id]
            if node.continuation_counts is None:
                node.continuation_counts = continuation_counts(node.phrase)
            return node.continuation_counts

        def candidate_score(node_id: int, byte: int) -> tuple[float, int, int, float]:
            node = nodes[node_id]
            counts = ensure_counts(node_id)
            support = int(counts[byte])
            residual = int(counts.sum()) - sum(
                int(counts[promoted]) for promoted in node.children
            )
            if support <= 0 or residual <= 0 or byte in node.children:
                return 0.0, support, residual, 0.0
            q = min(1.0, support / residual)
            entropy = _binary_entropy(q)
            score = residual * entropy if mode == "entropy" else float(support)
            return score, support, residual, entropy

        def push_candidates(node_id: int) -> None:
            nonlocal serial
            node = nodes[node_id]
            counts = ensure_counts(node_id)
            for byte in np.flatnonzero(counts):
                byte = int(byte)
                if byte in node.children:
                    continue
                score, support, _residual, _entropy = candidate_score(node_id, byte)
                heapq.heappush(
                    heap,
                    (-score, -support, serial, node_id, byte, node.version),
                )
                serial += 1

        for node_id in range(256):
            push_candidates(node_id)

        target_phrase_tokens = vocab_size - 1
        expansions: list[Expansion] = []
        next_token_id = 256

        while next_token_id < target_phrase_tokens:
            while heap:
                _neg_score, _neg_support, _serial, parent_id, byte, version = heapq.heappop(
                    heap
                )
                parent = nodes[parent_id]
                if version != parent.version or byte in parent.children:
                    continue
                score, support, residual, entropy = candidate_score(parent_id, byte)
                if support <= 0:
                    continue
                break
            else:
                raise RuntimeError(
                    f"ran out of supported prefix extensions at vocab size {next_token_id + 1}"
                )

            parent = nodes[parent_id]
            q = min(1.0, support / residual) if residual else 0.0
            child_phrase = parent.phrase + bytes([byte])
            child_id = len(nodes)
            nodes.append(_Node(phrase=child_phrase, token_id=next_token_id))
            parent.children[byte] = child_id
            parent.version += 1
            expansions.append(
                Expansion(
                    parent=parent.phrase,
                    child=child_phrase,
                    byte=byte,
                    child_occurrences=support,
                    residual_occurrences=residual,
                    q=q,
                    binary_entropy_bits=entropy,
                    score=score,
                )
            )
            next_token_id += 1

            # Adding one child changes the parent's residual branch, so refresh
            # its remaining candidate decisions. The new phrase can itself be
            # selectively refined one byte deeper.
            push_candidates(parent_id)
            push_candidates(child_id)

        return cls(nodes, root_children, vocab_size, expansions, mode)

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"), add_eos=add_eos)

    def encode_bytes(self, data: bytes, *, add_eos: bool = False) -> list[int]:
        out: list[int] = []
        pos = 0
        while pos < len(data):
            node_id = self.root_children[data[pos]]
            node = self.nodes[node_id]
            best_id = node.token_id
            best_end = pos + 1
            cursor = pos + 1

            while cursor < len(data):
                child_id = node.children.get(data[cursor])
                if child_id is None:
                    break
                node_id = child_id
                node = self.nodes[node_id]
                cursor += 1
                best_id = node.token_id
                best_end = cursor

            out.append(best_id)
            pos = best_end

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

    def token_piece(self, token_id: int) -> bytes:
        if token_id == self.eos_id:
            return b""
        return self._id_to_phrase[token_id]

    def max_phrase_bytes(self) -> int:
        return max(map(len, self._id_to_phrase))
