import math

import pytest
import torch
from torch import nn

from tokenizer_experiment.experiment import _group_tunstall_safe_rows
from tokenizer_experiment.model import CausalTransformer, ModelConfig
from tokenizer_experiment.prequential import _online_datum_step
from tokenizer_experiment.tunstall import EmpiricalTunstallTokenizer


class BiasOnlyModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        batch, time = ids.shape
        return self.bias.view(1, 1, -1).expand(batch, time, -1)


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr: float):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def test_online_datum_scores_before_one_update_across_context_chunks():
    vocab_size = 17
    model = BiasOnlyModel(vocab_size)
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    ids = [1, 2, 3, 4, 5, 6, 7]

    bits = _online_datum_step(
        model,
        optimizer,
        ids,
        eos_id=16,
        context=3,
        device=torch.device("cpu"),
    )

    # Content tokens plus one EOS termination target are all scored while the
    # model is still uniform, even though the datum spans several context chunks.
    assert bits == pytest.approx((len(ids) + 1) * math.log2(vocab_size), rel=1e-6)
    assert optimizer.step_calls == 1
    assert not torch.allclose(model.bias, torch.zeros_like(model.bias))


def test_empty_content_datum_still_predicts_eos_once():
    vocab_size = 7
    model = BiasOnlyModel(vocab_size)
    optimizer = CountingSGD(model.parameters(), lr=0.1)

    bits = _online_datum_step(
        model,
        optimizer,
        [],
        eos_id=6,
        context=4,
        device=torch.device("cpu"),
    )

    assert bits == pytest.approx(math.log2(vocab_size), rel=1e-6)
    assert optimizer.step_calls == 1


def test_tunstall_row_grouping_closes_at_every_earliest_legal_boundary():
    # With one 256-way expansion on an all-'a' corpus, the byte 'a' is expanded
    # and therefore needs a second byte before a Tunstall phrase completes.
    tokenizer = EmpiricalTunstallTokenizer.train(
        b"a" * 100,
        requested_vocab_size=512,
        mode="boundary",
    )
    rows = [b"a", b"a", b"a", b"a"]

    datums, rows_per_datum, dropped = _group_tunstall_safe_rows(rows, tokenizer)

    assert datums == [b"aa", b"aa"]
    assert rows_per_datum == [2, 2]
    assert dropped == 0
    for datum in datums:
        tokenizer.encode_bytes(datum)


def test_transformer_uses_small_tied_embedding_initialization():
    model = CausalTransformer(vocab_size=257, cfg=ModelConfig(d_model=64, n_heads=4))

    assert model.head.weight.data_ptr() == model.token.weight.data_ptr()
    assert float(model.token.weight.std()) == pytest.approx(0.02, abs=0.002)
