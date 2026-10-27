"""Flow-matching models for guided posterior sampling.

Notation
--------
theta : parameters of interest,  shape (B, d)
eps   : experimental designs,    shape (B, n, de)
y     : observations,            shape (B, n, dy)
t     : flow time in [0, 1], shape (B,); t = 0 is the Gaussian base, t = 1 is the prior p(theta).

Probability path (as in scRatio)
    theta_t = t * theta_1 + lambda_t * theta_0,   theta_0 ~ N(0, I)
    lambda_t = sqrt((1 - (1 - sigma_min) t)^2 + sigma t (1 - t)),   lambda_sp_t = lambda_t * d lambda_t / dt
    u_t      = theta_1 + lambda_sp_t / lambda_t * theta_0             (conditional target velocity)
    c_t      = lambda_t^2 - lambda_sp_t * t,   c_t * score = t * u_t - theta_t   (conditional score target)
The marginal velocity and score are linked by  v_t(x) = (x + c_t * s_t(x)) / t.

Guidance
    By Bayes' rule  grad log p_t(theta | y, eps) = grad log p_t(theta) + grad log p_t(y | theta, eps).
    ``PriorFlowModel`` learns v_t and s_t of the prior. ``LikelihoodGuidanceModel`` learns
    g ~= grad log p_t(y | theta, eps) by fitting  s_prior + g  to the posterior score, with g a sum of
    per-observation terms (log p(y_{1:n} | theta, eps_{1:n}) = sum_i log p(y_i | theta, eps_i)).
    Exactly, g = t / lambda_t^2 * (E[theta_1 | theta_t, y, eps] - E[theta_1 | theta_t]), and the posterior velocity is
    v_prior + c_t / t * g. The network learns the scheduled term  w = c_t / t * g  directly, so sampling needs no
    division by t, lambda_t or c_t:  v_post = v_prior + w.  In the score loss, c_t * g = t * w.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from flow_bed.models.history_encoder import HistoryEncoder


def timestep_embedding(t: Tensor, dim: int, max_period: float = 1e4) -> Tensor:
    """Sinusoidal embedding of times in [0, 1]. ``t``: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


ACTIVATIONS = {"selu": nn.SELU, "silu": nn.SiLU}
LEGACY_ACTIVATION = "silu"  # checkpoints saved before the activation option: load with cfg.get("activation", this)


class MLP(nn.Sequential):
    """``activation``: "silu" (default, smooth, so divergences can be differentiated again) or "selu"."""

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, activation: str = "silu"):
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), ACTIVATIONS[activation]()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        super().__init__(*layers)


class PriorFlowModel(nn.Module):
    """Flow matching model for the prior p(theta) that learns both the velocity and the score."""

    def __init__(
        self,
        dim: int,
        hidden_dims: list[int] = (256, 256, 256),
        time_embed_dim: int = 64,
        sigma_min: float = 0.0,
        sigma: float = 0.0,
        activation: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.time_embed_dim = time_embed_dim
        self.sigma_min = sigma_min
        self.sigma = sigma
        self.vf_net = MLP(dim + time_embed_dim, list(hidden_dims), dim, activation)
        self.score_net = MLP(dim + time_embed_dim, list(hidden_dims), dim, activation)

    def lambda_t(self, t: Tensor) -> Tensor:
        return torch.sqrt((1 - (1 - self.sigma_min) * t) ** 2 + self.sigma * t * (1 - t))

    def lambda_sp_t(self, t: Tensor) -> Tensor:
        """lambda_t * d lambda_t / dt."""
        return (self.sigma * (1 - 2 * t) - 2 * (1 - self.sigma_min) * (1 - (1 - self.sigma_min) * t)) / 2

    def forward(self, theta_t: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Velocity v_t and score s_t at ``theta_t`` (B, d), ``t`` (B,)."""
        h = torch.cat([theta_t, timestep_embedding(t, self.time_embed_dim)], dim=-1)
        return self.vf_net(h), self.score_net(h)

    def shared_step(self, theta1: Tensor) -> Tensor:
        theta0 = torch.randn_like(theta1)
        t = torch.rand(theta1.shape[0], 1, device=theta1.device)
        lam, lam_sp = self.lambda_t(t), self.lambda_sp_t(t)
        theta_t = t * theta1 + lam * theta0
        u_t = theta1 + lam_sp / lam * theta0
        c_t = lam**2 - lam_sp * t

        v, s = self(theta_t, t[:, 0])
        return F.mse_loss(v, u_t) + F.mse_loss(c_t * s, t * u_t - theta_t)


class LikelihoodGuidanceModel(nn.Module):
    """Learns w = c_t / t * grad log p_t(y | theta_t, eps) on top of a frozen, trained prior.

    The history {(y_i, eps_i)} enters through ``encoder`` (trained jointly with the guidance network). The posterior
    score  s_prior + g, g = t / c_t * w,  is trained with the same score loss as the prior, on joint samples
    (theta, eps, y) ~ p(theta) p(y | theta, eps) of shapes (B, d), (B, n, de), (B, n, dy).

    ``parametrization``: "w" - the network outputs w directly; "h" - it outputs h = g / t and w = c_t * h.
    """

    def __init__(
        self,
        prior: PriorFlowModel,
        encoder: HistoryEncoder,
        hidden_dims: list[int] = (256, 256, 256),
        time_embed_dim: int = 64,
        parametrization: str = "w",
        activation: str = "silu",
    ):
        super().__init__()
        self.prior = prior.requires_grad_(False)
        self.encoder = encoder
        self.time_embed_dim = time_embed_dim
        self.parametrization = parametrization
        self.net = MLP(prior.dim + time_embed_dim + encoder.out_dim, list(hidden_dims), prior.dim, activation)

    def forward(self, theta_t: Tensor, t: Tensor, history: Tensor) -> Tensor:
        """w -> (B, d), the term added to the prior velocity; ``history`` = encoder(y, eps), shape (B, out_dim)."""
        out = self.net(torch.cat([theta_t, timestep_embedding(t, self.time_embed_dim), history], dim=-1))
        if self.parametrization == "h":
            t_ = t[:, None]
            out = (self.prior.lambda_t(t_) ** 2 - self.prior.lambda_sp_t(t_) * t_) * out
        return out

    def w_and_h(self, theta_t: Tensor, t: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        """(w, h) with w = c_t h, from one network evaluation. With the "h" parametrisation h is the network output
        (no division, also near t = 1, where c_t -> 0); with "w", h = w / c_t."""
        out = self.net(torch.cat([theta_t, timestep_embedding(t, self.time_embed_dim), history], dim=-1))
        t_ = t[:, None]
        c_t = self.prior.lambda_t(t_) ** 2 - self.prior.lambda_sp_t(t_) * t_
        return (c_t * out, out) if self.parametrization == "h" else (out, out / c_t)

    def shared_step(self, batch: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        theta1, eps, y = batch
        theta0 = torch.randn_like(theta1)
        t = torch.rand(theta1.shape[0], 1, device=theta1.device)
        lam, lam_sp = self.prior.lambda_t(t), self.prior.lambda_sp_t(t)
        theta_t = t * theta1 + lam * theta0
        u_t = theta1 + lam_sp / lam * theta0
        c_t = lam**2 - lam_sp * t

        with torch.no_grad():
            _, s_prior = self.prior(theta_t, t[:, 0])
        w = self(theta_t, t[:, 0], self.encoder(y, eps))
        return F.mse_loss(c_t * s_prior + t * w, t * u_t - theta_t)
