import pytest

from tunstall import EmpiricalTunstallTokenizer


def aligned_prefix(tok: EmpiricalTunstallTokenizer, data: bytes) -> bytes:
    end = tok.align_utf8_boundaries(data, [1.0])[0]
    return data[:end]


def test_round_trip_and_prefix_free():
    data = (b"banana bandana\n" * 100) + "hello λ".encode()
    tok = EmpiricalTunstallTokenizer.train(data, 1024, mode="empirical")
    tok.assert_prefix_free()
    aligned = aligned_prefix(tok, data)
    ids = tok.encode_bytes(aligned)
    assert tok.decode_bytes(ids) == aligned
    assert tok.vocab_size == EmpiricalTunstallTokenizer.legal_vocab_size(1024)
    assert tok.eos_id == tok.phrase_vocab_size
    assert tok.vocab_size == tok.phrase_vocab_size + 1


def test_iid_round_trip():
    data = b"abracadabra" * 100
    tok = EmpiricalTunstallTokenizer.train(data, 1024, mode="iid")
    tok.assert_prefix_free()
    aligned = aligned_prefix(tok, data)
    assert tok.decode_bytes(tok.encode_bytes(aligned)) == aligned


def test_boundary_round_trip_and_vocab_arithmetic():
    data = b"the quick brown fox jumps over the lazy dog\n" * 300
    tok = EmpiricalTunstallTokenizer.train(data, 4096, mode="boundary")
    tok.assert_prefix_free()
    aligned = aligned_prefix(tok, data)
    ids = tok.encode_bytes(aligned)
    assert tok.decode_bytes(ids) == aligned
    # 15 full 256-way expansions: 256 + 15*255 phrase leaves, plus EOS.
    assert tok.phrase_vocab_size == 4081
    assert tok.vocab_size == 4082


def test_eos_is_separate_from_phrase_tree():
    data = b"abcabcabcabc" * 100
    tok = EmpiricalTunstallTokenizer.train(data, 1024, mode="boundary")
    aligned = aligned_prefix(tok, data)
    ids = tok.encode_bytes(aligned, add_eos=True)
    assert ids[-1] == tok.eos_id
    assert tok.decode_bytes(ids) == aligned
    assert all(tok.eos_id not in phrase for phrase in tok._id_to_phrase)


def test_unaligned_finite_message_requires_boundary_alignment():
    data = b"a" * 1000
    tok = EmpiricalTunstallTokenizer.train(data, 1024, mode="boundary")
    # Find a prefix that ends inside an expanded phrase.
    for end in range(1, len(data)):
        try:
            tok.encode_bytes(data[:end])
        except ValueError:
            with pytest.raises(ValueError):
                tok.encode_bytes(data[:end])
            break
    else:
        pytest.skip("constructed tree happened to align every tested prefix")


def test_aligned_prequential_cuts_make_every_block_independently_encodable():
    data = (("the λ quick brown fox\n" * 200).encode())
    tok = EmpiricalTunstallTokenizer.train(data, 4096, mode="boundary")
    cuts = tok.align_utf8_boundaries(data, [0.1, 0.25, 0.5, 1.0])

    start = 0
    for end in cuts:
        block = data[start:end]
        assert block.decode("utf-8").encode("utf-8") == block
        ids = tok.encode_bytes(block)
        assert tok.decode_bytes(ids) == block
        start = end
