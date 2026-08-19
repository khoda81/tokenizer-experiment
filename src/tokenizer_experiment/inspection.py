from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .tunstall import BPETokenizer, EmpiricalTunstallTokenizer


def _bytelevel_decoder() -> dict[str, int]:
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = visible[:]
    codepoints = visible[:]
    extra = 0
    for value in range(256):
        if value not in visible:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}


_BYTELEVEL_DECODER = _bytelevel_decoder()


def bytelevel_piece_bytes(piece: str) -> bytes:
    """Invert Hugging Face ByteLevel's reversible byte-to-unicode alphabet."""
    return bytes(_BYTELEVEL_DECODER[ch] for ch in piece)


def display_bytes(piece: bytes, *, limit: int = 48) -> str:
    if len(piece) > limit:
        piece = piece[:limit]
        suffix = "…"
    else:
        suffix = ""
    text = piece.decode("utf-8", errors="backslashreplace")
    return repr(text)[1:-1] + suffix


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def categorical_entropy(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / total
    return float(-(probs * np.log2(probs)).sum())


def emitted_token_rows(
    tokenizer: BPETokenizer | EmpiricalTunstallTokenizer,
    raw: bytes,
    *,
    top_n: int = 40,
) -> list[dict[str, Any]]:
    text = raw.decode("utf-8")
    ids = tokenizer.encode(text)
    counts = Counter(ids)
    total = len(ids)

    rows: list[dict[str, Any]] = []
    for token_id, count in counts.most_common(top_n):
        if isinstance(tokenizer, EmpiricalTunstallTokenizer):
            piece = bytes(tokenizer.token_piece(token_id))
        else:
            token_text = tokenizer.tokenizer.id_to_token(token_id)
            if token_text is None or token_id == tokenizer.eos_id:
                piece = b""
            else:
                piece = bytelevel_piece_bytes(token_text)
        rows.append(
            {
                "token_id": token_id,
                "count": count,
                "probability": count / total,
                "bytes": len(piece),
                "piece": display_bytes(piece),
                "hex": piece.hex(),
            }
        )
    return rows


def tunstall_split_rows(
    tokenizer: EmpiricalTunstallTokenizer,
    raw: bytes,
) -> list[dict[str, Any]]:
    """Measure the actual 256-way continuations encountered at expanded nodes."""
    continuation_counts: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(256, dtype=np.int64)
    )
    node_visits: Counter[int] = Counter()
    emitted_tokens = 0
    node_id = tokenizer.root

    for symbol in raw:
        # If we are below the root, the current node represents an expanded
        # prefix and `symbol` is the byte on which its 256-way branch is taken.
        if node_id != tokenizer.root:
            node = tokenizer.nodes[node_id]
            if not node.is_leaf:
                continuation_counts[node_id][symbol] += 1
                node_visits[node_id] += 1

        children = tokenizer.nodes[node_id].children
        if children is None:
            raise AssertionError(
                "expected internal node while inspecting Tunstall tree"
            )
        node_id = children[symbol]
        if tokenizer.nodes[node_id].is_leaf:
            emitted_tokens += 1
            node_id = tokenizer.root

    rows: list[dict[str, Any]] = []
    for current_id, node in enumerate(tokenizer.nodes):
        if current_id == tokenizer.root or node.is_leaf:
            continue
        counts = continuation_counts[current_id]
        visits = int(node_visits[current_id])
        entropy = categorical_entropy(counts)
        observed = int(np.count_nonzero(counts))
        top = np.argsort(counts)[::-1]
        continuations = []
        for symbol in top[:8]:
            count = int(counts[symbol])
            if count == 0:
                break
            continuations.append(
                {
                    "byte": int(symbol),
                    "piece": display_bytes(bytes([int(symbol)])),
                    "count": count,
                    "conditional_probability": count / visits if visits else 0.0,
                }
            )
        rows.append(
            {
                "prefix": display_bytes(bytes(node.phrase)),
                "prefix_hex": bytes(node.phrase).hex(),
                "prefix_bytes": len(node.phrase),
                "visits": visits,
                "mass_per_emitted_token": visits / emitted_tokens
                if emitted_tokens
                else 0.0,
                "observed_children": observed,
                "unused_children": 256 - observed,
                "next_byte_entropy_bits": entropy,
                "effective_branching": 2**entropy,
                "entropy_per_added_vocab_slot": entropy / 255.0,
                "top_continuations": continuations,
            }
        )

    rows.sort(key=lambda row: row["visits"], reverse=True)
    return rows


def _bpe_merges(tokenizer: BPETokenizer) -> list[tuple[str, str]]:
    model = json.loads(tokenizer.tokenizer.to_str())["model"]
    merges: list[tuple[str, str]] = []
    for merge in model.get("merges", []):
        if isinstance(merge, str):
            left, right = merge.split(" ", maxsplit=1)
        else:
            left, right = merge
        merges.append((left, right))
    return merges


def bpe_merge_split_rows(
    tokenizer: BPETokenizer,
    raw: bytes,
    *,
    min_left_occurrences: int = 100,
) -> list[dict[str, Any]]:
    """Approximate BPE merges as binary continuation tests on raw text.

    For merge A+B -> AB, q is the arbitrary-position corpus estimate
    P(B follows A | A occurs). This is deliberately a structural diagnostic,
    not a reconstruction of the BPE trainer's historical pair counts.
    """
    rows: list[dict[str, Any]] = []
    for rank, (left_text, right_text) in enumerate(_bpe_merges(tokenizer)):
        left = bytelevel_piece_bytes(left_text)
        right = bytelevel_piece_bytes(right_text)
        merged = left + right
        left_count = raw.count(left)
        if left_count < min_left_occurrences:
            continue
        merged_count = raw.count(merged)
        q = min(1.0, merged_count / left_count)
        rows.append(
            {
                "merge_rank": rank,
                "left": display_bytes(left),
                "right": display_bytes(right),
                "merged": display_bytes(merged),
                "left_bytes": len(left),
                "right_bytes": len(right),
                "merged_bytes": len(merged),
                "left_occurrences": left_count,
                "merged_occurrences": merged_count,
                "q_followed_by_right": q,
                "binary_entropy_bits": binary_entropy(q),
            }
        )
    return rows


def summarize_bpe_splits(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    entropies = np.asarray(
        [row["binary_entropy_bits"] for row in rows], dtype=np.float64
    )
    qs = np.asarray([row["q_followed_by_right"] for row in rows], dtype=np.float64)
    support = np.asarray([row["left_occurrences"] for row in rows], dtype=np.float64)
    return {
        "count": float(len(rows)),
        "mean_binary_entropy_bits": float(entropies.mean()),
        "median_binary_entropy_bits": float(np.median(entropies)),
        "support_weighted_binary_entropy_bits": float(
            np.average(entropies, weights=support)
        ),
        "median_q": float(np.median(qs)),
        "fraction_q_between_0.25_and_0.75": float(np.mean((qs >= 0.25) & (qs <= 0.75))),
    }
