from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    context: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0


class CausalTransformer(nn.Module):
    """Small decoder-only Transformer used by the tokenizer experiments.

    This intentionally preserves the architecture used for the first documented
    BPE-vs-Tunstall baseline. Keep architecture changes separate from experiment
    plumbing changes so historical results remain interpretable.
    """

    def __init__(self, vocab_size: int, cfg: ModelConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.context = cfg.context
        self.token = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.mlp_ratio,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        # norm_first=True makes PyTorch disable the nested-tensor fast path
        # anyway. Set this explicitly so constructing each prequential model
        # does not emit the same warning while preserving the baseline model.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # nn.Embedding defaults to N(0, 1), which is disastrous after tying it
        # to the output projection. Start near a uniform softmax instead.
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
        self.head.weight = self.token.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        _, t = ids.shape
        if t > self.context:
            raise ValueError(f"sequence length {t} exceeds context {self.context}")
        positions = torch.arange(t, device=ids.device)
        h = self.token(ids) + self.pos(positions)[None, :, :]
        mask = torch.triu(
            torch.ones(t, t, device=ids.device, dtype=torch.bool), diagonal=1
        )
        h = self.transformer(h, mask=mask)
        return self.head(self.norm(h))
