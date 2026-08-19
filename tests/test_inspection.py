import pytest

from tokenizer_experiment.inspection import (
    binary_entropy,
    count_overlapping,
    tunstall_split_rows,
)
from tokenizer_experiment.tunstall import EmpiricalTunstallTokenizer


def test_binary_entropy():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.25) == pytest.approx(binary_entropy(0.75))


def test_count_overlapping_occurrences():
    assert count_overlapping(b"aaaaa", b"aa") == 4
    assert count_overlapping(b"banana", b"ana") == 2


def test_tunstall_inspection_reports_one_row_per_expansion():
    data = b"the quick brown fox jumps over the lazy dog\n" * 300
    tok = EmpiricalTunstallTokenizer.train(data, 4096, mode="boundary")
    end = tok.align_utf8_boundaries(data, [1.0])[0]
    rows = tunstall_split_rows(tok, data[:end])

    expected_expansions = (tok.phrase_vocab_size - 256) // 255
    assert len(rows) == expected_expansions
    assert all(0 <= row["observed_children"] <= 256 for row in rows)
    assert all(0.0 <= row["next_byte_entropy_bits"] <= 8.0 for row in rows)
    assert all(
        row["entropy_per_added_vocab_slot"]
        == pytest.approx(row["next_byte_entropy_bits"] / 255.0)
        for row in rows
    )
