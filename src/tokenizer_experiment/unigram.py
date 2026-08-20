from __future__ import annotations

import math
from collections.abc import Iterable


class ByteUnigramTokenizer:
    """Byte-complete Unigram LM tokenizer using Hugging Face Tokenizers.

    The tokenizer is trained on ByteLevel's reversible 256-symbol alphabet, so
    every UTF-8 byte string remains representable without an unknown token.
    Unlike BPE, vocabulary selection is driven by a unigram language-model
    likelihood objective with EM/pruning.

    One extra vocabulary entry is reserved as BOS for the Transformer. It never
    appears in raw-stream tokenization.
    """

    BOS_TOKEN = "<|bos|>"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        bos = tokenizer.token_to_id(self.BOS_TOKEN)
        if bos is None:
            raise RuntimeError("Unigram tokenizer has no reserved BOS token")
        self.eos_id = bos  # historical model interface name; used as BOS only.

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        *,
        max_piece_length: int = 16,
        shrinking_factor: float = 0.75,
        n_sub_iterations: int = 2,
    ) -> ByteUnigramTokenizer:
        if max_piece_length <= 0:
            raise ValueError("max_piece_length must be positive")
        if not 0.0 < shrinking_factor < 1.0:
            raise ValueError("shrinking_factor must be in (0, 1)")
        if n_sub_iterations <= 0:
            raise ValueError("n_sub_iterations must be positive")

        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.Unigram())
        # No Unicode normalization: this experiment codes the exact UTF-8 bytes.
        # use_regex=False keeps the input as one byte-level stream instead of
        # imposing GPT-2-style word/whitespace pre-tokenization boundaries.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=False
        )
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            special_tokens=[cls.BOS_TOKEN],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            shrinking_factor=shrinking_factor,
            unk_token=None,
            max_piece_length=max_piece_length,
            n_sub_iterations=n_sub_iterations,
        )

        # Iterator chunks are only a trainer-memory boundary. The fitted model is
        # later applied once to the complete continuous evaluation stream.
        char_chunk = 8_192
        iterator = (text[i : i + char_chunk] for i in range(0, len(text), char_chunk))
        tokenizer.train_from_iterator(
            iterator, trainer=trainer, length=math.ceil(len(text) / char_chunk)
        )
        wrapped = cls(tokenizer)
        if wrapped.vocab_size != vocab_size:
            raise RuntimeError(
                f"Unigram trainer produced vocab={wrapped.vocab_size}, expected {vocab_size}. "
                "Try a larger tokenizer-fit prefix or a smaller vocabulary."
            )
        return wrapped

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def token_piece(self, token_id: int) -> tuple[int, ...]:
        if token_id == self.eos_id:
            return ()
        piece = self.tokenizer.id_to_token(token_id)
        if piece is None:
            raise ValueError(f"unknown Unigram token id: {token_id}")
        return tuple(_bytelevel_piece_bytes(piece))

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        out = bytearray()
        for token_id in ids:
            if token_id == self.eos_id:
                continue
            out.extend(self.token_piece(token_id))
        return bytes(out)

    def max_phrase_bytes(self) -> int:
        return max(
            len(self.token_piece(token_id))
            for token_id in range(self.vocab_size)
            if token_id != self.eos_id
        )


def _bytelevel_piece_bytes(piece: str) -> bytes:
    """Invert the reversible byte-to-Unicode alphabet used by ByteLevel."""
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
    decoder = {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}
    return bytes(decoder[ch] for ch in piece)
