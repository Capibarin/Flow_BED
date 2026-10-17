"""Order-invariant encoders of an experimental history {(y_i, eps_i)}_{i=1..n}.

``HistoryEncoder``              DeepSets: e_i = phi(y_i, eps_i), pooled = mean_i e_i.
``SelfAttentionHistoryEncoder`` e_i = phi(y_i, eps_i), then self-attention layers without positional encoding (so the
                                observations interact, and the result stays permutation equivariant), then the mean.
Both append log(1 + n) so that the number of experiments is known, and encode an empty history (n = 0) as a zero
embedding with log(1 + 0) = 0. Output shape (B, embed_dim + 1) = (B, out_dim).
Observations are standardised as (y - y_loc) / y_scale before phi (identity by default).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from flow_bed.models.flow_matching import MLP


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        y_dim: int,
        eps_dim: int,
        hidden_dims: list[int] = (256, 256, 256),
        embed_dim: int = 64,
        y_loc: float = 0.0,
        y_scale: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = embed_dim + 1
        self.y_loc, self.y_scale = y_loc, y_scale
        self.phi = MLP(y_dim + eps_dim, list(hidden_dims), embed_dim)

    def forward(self, y: Tensor, eps: Tensor) -> Tensor:
        """``y`` (B, n, dy), ``eps`` (B, n, de) -> (B, embed_dim + 1)."""
        B, n, _ = y.shape
        if n == 0:
            pooled = y.new_zeros(B, self.embed_dim)
        else:
            pooled = self.phi(torch.cat([(y - self.y_loc) / self.y_scale, eps], dim=-1)).mean(dim=1)
        return torch.cat([pooled, y.new_full((B, 1), math.log1p(n))], dim=-1)


class SelfAttentionHistoryEncoder(nn.Module):
    def __init__(
        self,
        y_dim: int,
        eps_dim: int,
        hidden_dims: list[int] = (256, 256, 256),
        embed_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        y_loc: float = 0.0,
        y_scale: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = embed_dim + 1
        self.y_loc, self.y_scale = y_loc, y_scale
        self.phi = MLP(y_dim + eps_dim, list(hidden_dims), embed_dim)
        layer = nn.TransformerEncoderLayer(
            embed_dim, n_heads, dim_feedforward=2 * embed_dim, dropout=0.0, batch_first=True, norm_first=True
        )
        self.attention = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, y: Tensor, eps: Tensor) -> Tensor:
        """``y`` (B, n, dy), ``eps`` (B, n, de) -> (B, embed_dim + 1)."""
        B, n, _ = y.shape
        if n == 0:
            pooled = y.new_zeros(B, self.embed_dim)
        else:
            e = self.phi(torch.cat([(y - self.y_loc) / self.y_scale, eps], dim=-1))  # (B, n, embed_dim)
            pooled = self.attention(e).mean(dim=1)
        return torch.cat([pooled, y.new_full((B, 1), math.log1p(n))], dim=-1)
