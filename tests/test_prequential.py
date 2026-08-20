import math

import pytest
import torch
from torch import nn

from tokenizer_experiment.experiment import _shared_update_boundaries
from tokenizer_experiment.model import CausalTransformer, ModelConfig
from tokenizer_experiment.prequential import _stream_update_step, _tail_rates


class RecordingBiasModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.inputs: list[list[int]] = []

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        self.inputs.append(ids[0].detach().cpu().tolist())
        batch, time = ids.shape
        return self.bias.view(1, 1, -1).expand(batch, time, -1)


class NextTokenModel(nn.Module):
    """Predict token x+1 from input token x, exposing alignment mistakes."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        predicted = (ids + 1) % self.vocab_size
        logits = torch.zeros(
            *ids.shape, self.vocab_size, dtype=torch.float32, device=ids.device
        )
        logits.scatter_(-1, predicted[..., None], 10.0)
        # Keep a parameter in the graph so the normal backward/update path runs.
        return logits + self.dummy * 0.0


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr: float):
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def test_stream_update_scores_before_one_step_and_keeps_prior_context():
    vocab_size = 17
    model = RecordingBiasModel(vocab_size)
    optimizer = CountingSGD(model.parameters(), lr=0.0)
    ids = [1, 2, 3, 4]

    first_bits = _stream_update_step(
        model,
        optimizer,
        ids,
        start_token=0,
        end_token=2,
        bos_id=16,
        context=4,
        device=torch.device("cpu"),
    )
    second_bits = _stream_update_step(
        model,
        optimizer,
        ids,
        start_token=2,
        end_token=4,
        bos_id=16,
        context=4,
        device=torch.device("cpu"),
    )

    assert first_bits == pytest.approx(2 * math.log2(vocab_size), rel=1e-6)
    assert second_bits == pytest.approx(2 * math.log2(vocab_size), rel=1e-6)
    assert optimizer.step_calls == 2

    # The second update is not a new sample. Its input window includes tokens
    # from the first update; BOS appears only because the whole stream is still
    # shorter than the model context.
    assert model.inputs[0] == [16, 1]
    assert model.inputs[1] == [16, 1, 2, 3]


def test_stream_targets_are_aligned_with_the_immediately_previous_token():
    # BOS=0 predicts 1, then 1 predicts 2, ..., so the correctly shifted stream
    # should receive the model's high-probability class at every target. This
    # catches an off-by-one even though the uniform-bias tests cannot.
    vocab_size = 8
    ids = [1, 2, 3, 4, 5, 6]
    model = NextTokenModel(vocab_size)
    optimizer = CountingSGD(model.parameters(), lr=0.0)

    bits = _stream_update_step(
        model,
        optimizer,
        ids,
        start_token=0,
        end_token=len(ids),
        bos_id=0,
        context=4,
        device=torch.device("cpu"),
    )

    correct_nats = math.log(math.exp(10.0) + vocab_size - 1) - 10.0
    expected_bits = len(ids) * correct_nats / math.log(2.0)
    assert bits == pytest.approx(expected_bits, rel=1e-4, abs=1e-6)
    assert optimizer.step_calls == 1


def test_large_update_accumulates_chunks_before_single_step():
    vocab_size = 11
    model = RecordingBiasModel(vocab_size)
    optimizer = CountingSGD(model.parameters(), lr=0.0)
    ids = [i % 10 for i in range(9)]

    bits = _stream_update_step(
        model,
        optimizer,
        ids,
        start_token=0,
        end_token=len(ids),
        bos_id=10,
        context=4,
        device=torch.device("cpu"),
    )

    assert bits == pytest.approx(len(ids) * math.log2(vocab_size), rel=1e-6)
    assert len(model.inputs) > 1
    assert optimizer.step_calls == 1


def test_shared_update_boundaries_follow_raw_milestones():
    offsets = {
        "a": [0, 100, 256, 300, 512, 768, 1000],
        "b": [0, 128, 256, 400, 512, 700, 768, 1000],
    }

    assert _shared_update_boundaries(offsets, 250) == [256, 512, 768, 1000]


def test_tail_rates_use_nearest_real_optimizer_boundary():
    # Cumulative records correspond to stream start plus four update boundaries.
    raw = [0, 100, 260, 510, 800]
    bits = [0.0, 300.0, 620.0, 1020.0, 1455.0]

    tail = _tail_rates(raw, bits, windows=(300, 500))

    # For a requested 300-byte tail, target start=500 and boundary 510 is closest.
    assert tail["300"]["actual_bytes"] == 290
    assert tail["300"]["bits_per_byte"] == pytest.approx((1455 - 1020) / 290)
    # For requested 500 bytes, target start=300 and boundary 260 is closest.
    assert tail["500"]["actual_bytes"] == 540
    assert tail["500"]["bits_per_byte"] == pytest.approx((1455 - 620) / 540)


def test_transformer_uses_small_tied_embedding_initialization():
    model = CausalTransformer(vocab_size=257, cfg=ModelConfig(d_model=64, n_heads=4))

    assert model.head.weight.data_ptr() == model.token.weight.data_ptr()
    assert model.token.weight.detach().std().item() == pytest.approx(0.02, abs=0.002)
