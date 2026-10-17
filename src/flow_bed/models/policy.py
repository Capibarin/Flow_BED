"""Design policy: proposes the next design eps_{n+1} from the encoded history {(y_i, eps_i)}_{i=1..n}."""

from __future__ import annotations

from torch import Tensor, nn

from flow_bed.models.flow_matching import MLP
from flow_bed.models.history_encoder import HistoryEncoder


class Policy(nn.Module):
    def __init__(self, encoder: HistoryEncoder, eps_dim: int, hidden_dims: list[int] = (256, 256, 256)):
        super().__init__()
        self.encoder = encoder
        self.net = MLP(encoder.out_dim, list(hidden_dims), eps_dim)

    def forward(self, y: Tensor, eps: Tensor) -> Tensor:
        """``y`` (B, n, dy), ``eps`` (B, n, de) with n >= 0 -> next design (B, de)."""
        return self.net(self.encoder(y, eps))
