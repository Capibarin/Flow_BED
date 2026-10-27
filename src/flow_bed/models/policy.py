"""Design policy: proposes the next design eps_{n+1} from the encoded history {(y_i, eps_i)}_{i=1..n}.

``design_bound`` s (optional) keeps designs bounded, so the policy <-> encoder loop cannot leave the range the networks
were trained on: "radial" maps u -> s tanh(|u| / s) u / |u| (|eps| < s), "coord" maps each coordinate u -> s tanh(u / s).
Both are the identity near 0. Without a bound (default, as all checkpoints saved before the option) eps = net(h).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flow_bed.models.flow_matching import MLP
from flow_bed.models.history_encoder import HistoryEncoder


class Policy(nn.Module):
    def __init__(
        self,
        encoder: HistoryEncoder,
        eps_dim: int,
        hidden_dims: list[int] = (256, 256, 256),
        activation: str = "silu",
        design_bound: float | None = None,
        bound_type: str = "radial",
    ):
        super().__init__()
        self.encoder = encoder
        self.net = MLP(encoder.out_dim, list(hidden_dims), eps_dim, activation)
        self.design_bound = design_bound
        self.bound_type = bound_type

    def design(self, history: Tensor) -> Tensor:
        """Encoded history (B, out_dim) -> next design (B, de), bounded if ``design_bound`` is set."""
        u = self.net(history)
        s = self.design_bound
        if s is None:
            return u
        if self.bound_type == "coord":
            return s * torch.tanh(u / s)
        norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return s * torch.tanh(norm / s) * u / norm

    def forward(self, y: Tensor, eps: Tensor) -> Tensor:
        """``y`` (B, n, dy), ``eps`` (B, n, de) with n >= 0 -> next design (B, de)."""
        return self.design(self.encoder(y, eps))
