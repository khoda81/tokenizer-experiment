from __future__ import annotations


def mb(n: int) -> float:
    return n / 1_000_000


def safe_prefix(raw: bytes, max_bytes: int) -> tuple[bytes, bytes]:
    """Split bytes at or before max_bytes without cutting a UTF-8 codepoint."""
    if max_bytes <= 0 or max_bytes >= len(raw):
        return raw, b""
    n = max_bytes
    while n > 0:
        try:
            prefix = raw[:n].decode("utf-8")
            return prefix.encode("utf-8"), raw[n:]
        except UnicodeDecodeError as exc:
            if exc.end == n:
                n -= 1
            else:
                raise
    raise RuntimeError("could not find UTF-8 boundary")


def split_tokenizer_fit(raw: bytes, fit_bytes: int) -> tuple[bytes, bytes]:
    if fit_bytes >= len(raw):
        raise ValueError("tokenizer-fit prefix consumes the whole dataset")
    prefix, remainder = safe_prefix(raw, fit_bytes)
    if not remainder:
        raise ValueError("no bytes remain after tokenizer-fit prefix")
    return prefix, remainder
