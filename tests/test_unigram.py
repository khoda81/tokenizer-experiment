from tokenizer_experiment.unigram import ByteUnigramTokenizer


def test_byte_unigram_roundtrips_utf8_and_reserves_bos():
    text = (
        "hello hello world\n"
        "the quick brown fox jumps over the lazy dog\n"
        "café naïve résumé — 世界\n"
    ) * 300

    tokenizer = ByteUnigramTokenizer.train(
        text,
        vocab_size=300,
        max_piece_length=8,
    )

    sample = "hello, 世界 — café\n"
    ids = tokenizer.encode(sample)

    assert tokenizer.vocab_size == 300
    assert tokenizer.eos_id not in ids
    assert tokenizer.decode_bytes(ids) == sample.encode("utf-8")
    assert tokenizer.max_phrase_bytes() <= 8


def test_byte_unigram_can_encode_bytes_unseen_as_text_characters():
    # ByteLevel's full initial alphabet should retain every one of the 256 byte
    # symbols even when the training text is plain ASCII. Exercise that through
    # valid UTF-8 whose multibyte constituents were absent from training.
    tokenizer = ByteUnigramTokenizer.train(
        "ascii training text with repeated repeated phrases\n" * 400,
        vocab_size=280,
        max_piece_length=8,
    )

    sample = "λ🙂é"
    ids = tokenizer.encode(sample)

    assert tokenizer.decode_bytes(ids) == sample.encode("utf-8")
