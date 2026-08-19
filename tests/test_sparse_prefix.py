from tokenizer_experiment.inspection import display_bytes
from tokenizer_experiment.sparse_prefix import SparsePrefixTokenizer


def test_sparse_prefix_round_trip_and_exact_vocab_size():
    data = (b"the quick brown fox jumps over the lazy dog\n" * 50)
    tokenizer = SparsePrefixTokenizer.train(data, vocab_size=300, mode="entropy")

    ids = tokenizer.encode_bytes(data)
    assert tokenizer.decode_bytes(ids) == data
    assert tokenizer.vocab_size == 300
    assert tokenizer.eos_id == 299
    assert len(tokenizer.expansions) == 300 - 257


def test_promoting_child_keeps_parent_as_valid_token():
    data = b"ababababababacacacacac" * 20
    tokenizer = SparsePrefixTokenizer.train(data, vocab_size=270, mode="frequency")

    for expansion in tokenizer.expansions:
        parent_ids = [
            node.token_id for node in tokenizer.nodes if node.phrase == expansion.parent
        ]
        child_ids = [
            node.token_id for node in tokenizer.nodes if node.phrase == expansion.child
        ]
        assert parent_ids
        assert child_ids


def test_visible_whitespace_rendering():
    assert display_bytes(b" a\tb\nc\r") == "␠a⇥b⏎c␍"
