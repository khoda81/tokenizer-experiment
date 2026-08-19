from tunstall import EmpiricalTunstallTokenizer


def test_round_trip_and_prefix_free():
    data = (b"banana bandana\n" * 100) + "hello λ".encode()
    tok = EmpiricalTunstallTokenizer.train(data, 1024, mode="empirical")
    tok.assert_prefix_free()
    ids = tok.encode_bytes(data)
    assert tok.decode_bytes(ids) == data
    assert tok.vocab_size == EmpiricalTunstallTokenizer.legal_vocab_size(1024)


def test_iid_round_trip():
    data = b"abracadabra" * 100
    tok = EmpiricalTunstallTokenizer.train(data, 1024, mode="iid")
    tok.assert_prefix_free()
    assert tok.decode_bytes(tok.encode_bytes(data)) == data


def test_boundary_round_trip():
    data = b"the quick brown fox jumps over the lazy dog\n" * 300
    tok = EmpiricalTunstallTokenizer.train(data, 4096, mode="boundary")
    tok.assert_prefix_free()
    ids = tok.encode_bytes(data)
    assert tok.decode_bytes(ids) == data
    assert tok.vocab_size == 4097
