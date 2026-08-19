import math

import pytest
import torch
from torch import nn

from tokenizer_experiment.model import CausalTransformer, ModelConfig
from tokenizer_experiment.prequential import TokenWindows, score_model


def test_token_windows_score_each_token_after_first_exactly_once():
    ids = list(range(23))
    ds = TokenWindows(ids, context=5)

    targets = []
    for _x, y in ds:
        targets.extend(y.tolist())

    assert targets == ids[1:]


class UniformModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.anchor = nn.Parameter(torch.empty(0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        batch, time = ids.shape
        return torch.zeros(batch, time, self.vocab_size, device=ids.device)


def test_scoring_uniform_model_is_exact_uniform_code():
    vocab_size = 17
    ids = [i % vocab_size for i in range(37)]
    model = UniformModel(vocab_size)

    bits, _ = score_model(
        model,
        ids,
        context=7,
        device=torch.device("cpu"),
        batch_size=3,
    )

    assert bits == pytest.approx(len(ids) * math.log2(vocab_size), rel=1e-6)


def test_single_token_block_uses_uniform_code():
    vocab_size = 17
    model = UniformModel(vocab_size)

    bits, seconds = score_model(
        model,
        [3],
        context=7,
        device=torch.device("cpu"),
        batch_size=3,
    )

    assert bits == pytest.approx(math.log2(vocab_size))
    assert seconds == 0.0


def test_transformer_uses_small_tied_embedding_initialization():
    model = CausalTransformer(vocab_size=257, cfg=ModelConfig(d_model=64, n_heads=4))

    assert model.head.weight.data_ptr() == model.token.weight.data_ptr()
    assert float(model.token.weight.std()) == pytest.approx(0.02, abs=0.002)
