"""Flow-matching models for guided posterior sampling.

Notation
--------
theta : parameters of interest,            shape (B, d)
eps   : experimental design,               shape (B, de) or (B, n, de)
y     : observations,                      shape (B, dy) or (B, n, dy)
t     : flow time in [0, 1]; t = 0 is the Gaussian base, t = 1 is the data (prior) distribution.

Probability path (linear / rectified flow, optional ``sigma_min`` as in OT-CFM)
-----------------------------------------------------------------------------
    theta_t = alpha_t * theta_1 + sigma_t * theta_0,   theta_0 ~ N(0, I)
    alpha_t = t,   sigma_t = 1 - (1 - sigma_min) * t
    u_t     = theta_1 - (1 - sigma_min) * theta_0      (conditional target velocity)

Score <-> velocity for this Gaussian path (Tweedie):
    v_t(x) = x / t + c(t) * s_t(x),      c(t) = sigma_t / t,     s_t = grad_x log p_t(x)
    s_t(x) = (t * v_t(x) - x) / sigma_t

Guidance
--------
By Bayes' rule, grad_x log p_t(x | y, eps) = grad_x log p_t(x) + grad_x log p_t(y | x, eps), so the
conditional velocity is

    v_t(x | y, eps) = v_t(x) + c(t) * grad_x log p_t(y | x, eps).

``PriorFlowModel`` learns v_t(x); ``LikelihoodGuidanceModel`` learns g(x, t, y, eps) ~= grad_x log p_t(y | x, eps)
so that ``v_prior + c(t) * g`` reproduces the conditional flow. For n conditionally independent
observations, log p(y_{1:n} | theta, eps_{1:n}) = sum_i log p(y_i | theta, eps_i), hence the guidance
term is summed over the observation axis (exact at t = 1, the usual compositional approximation for t < 1).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor, nn

# --------------------------------------------------------------------------------------
# Small building blocks
# --------------------------------------------------------------------------------------


def timestep_embedding(t: Tensor, dim: int, max_period: float = 1e4) -> Tensor:
    """Sinusoidal embedding of scalar times in [0, 1]. ``t``: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t[:, None] * 1000.0 * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 256,
        n_layers: int = 4,
        act: Callable[[], nn.Module] = nn.SiLU,
    ):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1)
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), act()]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _as_batch_time(t: Tensor | float, batch: int, ref: Tensor) -> Tensor:
    """Broadcast scalar / (B,) / (B,1) time to shape (B,)."""
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), device=ref.device, dtype=ref.dtype)
    t = t.to(device=ref.device, dtype=ref.dtype)
    if t.numel() == 1:
        return t.reshape(()).expand(batch)
    return t.reshape(batch)


# --------------------------------------------------------------------------------------
# Probability path
# --------------------------------------------------------------------------------------


class LinearPath:
    """Linear interpolation path between N(0, I) at t=0 and the data at t=1."""

    def __init__(self, sigma_min: float = 0.0):
        self.sigma_min = float(sigma_min)

    def alpha(self, t: Tensor) -> Tensor:
        return t

    def sigma(self, t: Tensor) -> Tensor:
        return 1.0 - (1.0 - self.sigma_min) * t

    def interpolate(self, theta1: Tensor, theta0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return (theta_t, u_t) for t of shape (B,)."""
        t_ = t.reshape(-1, *([1] * (theta1.ndim - 1)))
        theta_t = self.alpha(t_) * theta1 + self.sigma(t_) * theta0
        u_t = theta1 - (1.0 - self.sigma_min) * theta0
        return theta_t, u_t

    def score_to_velocity_scale(self, t: Tensor, t_clip: float = 1e-3) -> Tensor:
        """c(t) = sigma_t / t such that  v_cond = v_prior + c(t) * grad log p_t(y | x)."""
        t = t.clamp(min=t_clip)
        return self.sigma(t) / t

    def velocity_to_score(self, x: Tensor, v: Tensor, t: Tensor, t_clip: float = 1e-3) -> Tensor:
        """s_t(x) = (t v - x) / sigma_t."""
        t_ = t.reshape(-1, *([1] * (x.ndim - 1)))
        sig = self.sigma(t_).clamp(min=t_clip)
        return (t_ * v - x) / sig


# --------------------------------------------------------------------------------------
# Prior flow  p(theta)
# --------------------------------------------------------------------------------------


class PriorFlowModel(nn.Module):
    """Unconditional flow matching model for the prior p(theta).

    ``forward(theta_t, t)`` returns the velocity field v_t(theta_t).
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 256,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        sigma_min: float = 0.0,
        act: Callable[[], nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.dim = dim
        self.time_embed_dim = time_embed_dim
        self.path = LinearPath(sigma_min)
        self.net = MLP(dim + time_embed_dim, dim, hidden, n_layers, act)

    def forward(self, theta_t: Tensor, t: Tensor | float) -> Tensor:
        tb = _as_batch_time(t, theta_t.shape[0], theta_t)
        h = torch.cat([theta_t, timestep_embedding(tb, self.time_embed_dim)], dim=-1)
        return self.net(h)

    def score(self, theta_t: Tensor, t: Tensor | float) -> Tensor:
        """grad log p_t(theta_t) implied by the learned velocity."""
        tb = _as_batch_time(t, theta_t.shape[0], theta_t)
        return self.path.velocity_to_score(theta_t, self(theta_t, tb), tb)

    def sample_time(self, batch: int, ref: Tensor, t_min: float = 0.0, t_max: float = 1.0) -> Tensor:
        return torch.rand(batch, device=ref.device, dtype=ref.dtype) * (t_max - t_min) + t_min

    def loss(self, theta1: Tensor, t: Optional[Tensor] = None, theta0: Optional[Tensor] = None) -> Tensor:
        """Conditional flow matching loss  E || v(theta_t, t) - u_t ||^2  on prior samples ``theta1``."""
        if theta0 is None:
            theta0 = torch.randn_like(theta1)
        if t is None:
            t = self.sample_time(theta1.shape[0], theta1)
        theta_t, u_t = self.path.interpolate(theta1, theta0, t)
        return ((self(theta_t, t) - u_t) ** 2).sum(-1).mean()


# --------------------------------------------------------------------------------------
# Likelihood guidance  grad_theta log p(y | theta, eps)
# --------------------------------------------------------------------------------------


class LikelihoodGuidanceModel(nn.Module):
    """Learns g(theta_t, t, y, eps) ~= grad_{theta_t} log p_t(y | theta_t, eps).

    * ``forward``             : the score-space correction, summed over observations -> (B, d)
    * ``velocity_correction`` : c(t) * forward(...), what gets added to the prior velocity
    * ``loss``                : trains  v_prior(theta_t, t) + c(t) g  to match the conditional CFM target
                               u_t on joint samples (theta, eps, y) ~ p(theta) p(y | theta, eps).

    Multiple observations
    ---------------------
    ``y`` may be (B, dy) or (B, n, dy) (likewise ``eps``). Per-observation terms are computed with a shared
    network and summed over n (optionally masked), i.e. the factorisation
    log p(y_{1:n} | theta, eps_{1:n}) = sum_i log p(y_i | theta, eps_i). Training with a random number of
    observations per batch is supported via ``mask``.
    """

    def __init__(
        self,
        dim: int,
        y_dim: int,
        eps_dim: int = 0,
        hidden: int = 256,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        sigma_min: float = 0.0,
        t_clip: float = 1e-3,
        act: Callable[[], nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.dim, self.y_dim, self.eps_dim = dim, y_dim, eps_dim
        self.time_embed_dim = time_embed_dim
        self.t_clip = t_clip
        self.path = LinearPath(sigma_min)
        self.net = MLP(dim + time_embed_dim + y_dim + eps_dim, dim, hidden, n_layers, act)

    # ---- shape helpers -------------------------------------------------------------
    @staticmethod
    def _to_obs_batch(x: Optional[Tensor], batch: int, feat: int, name: str) -> Optional[Tensor]:
        if x is None:
            if feat > 0:
                raise ValueError(f"`{name}` is required (dim {feat}) but got None")
            return None
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3 or x.shape[0] != batch or x.shape[-1] != feat:
            raise ValueError(f"`{name}` must have shape (B, dy) or (B, n, dy) with B={batch}, dy={feat}; got {tuple(x.shape)}")
        return x

    def per_observation(
        self,
        theta_t: Tensor,
        t: Tensor | float,
        y: Tensor,
        eps: Optional[Tensor] = None,
    ) -> Tensor:
        """Per-observation terms g_i(theta_t, t, y_i, eps_i) -> (B, n, d)."""
        B = theta_t.shape[0]
        y = self._to_obs_batch(y, B, self.y_dim, "y")
        eps = self._to_obs_batch(eps, B, self.eps_dim, "eps")
        n = y.shape[1]
        tb = _as_batch_time(t, B, theta_t)
        temb = timestep_embedding(tb, self.time_embed_dim)

        parts = [theta_t[:, None, :].expand(B, n, self.dim), temb[:, None, :].expand(B, n, self.time_embed_dim), y]
        if eps is not None:
            parts.append(eps.expand(B, n, self.eps_dim))
        h = torch.cat(parts, dim=-1)
        return self.net(h.reshape(B * n, -1)).reshape(B, n, self.dim)

    def forward(
        self,
        theta_t: Tensor,
        t: Tensor | float,
        y: Tensor,
        eps: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """sum_i g_i(theta_t, t, y_i, eps_i) -> (B, d). ``mask``: optional (B, n) bool/float."""
        g = self.per_observation(theta_t, t, y, eps)
        if mask is not None:
            g = g * mask.to(g.dtype)[..., None]
        return g.sum(dim=1)

    def velocity_correction(
        self,
        theta_t: Tensor,
        t: Tensor | float,
        y: Tensor,
        eps: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """c(t) * grad log p_t(y_{1:n} | theta_t, eps) — add this to the prior velocity."""
        tb = _as_batch_time(t, theta_t.shape[0], theta_t)
        c = self.path.score_to_velocity_scale(tb, self.t_clip)[:, None]
        return c * self(theta_t, tb, y, eps, mask)

    # ---- training --------------------------------------------------------------------
    def loss(
        self,
        prior: PriorFlowModel,
        theta1: Tensor,
        y: Tensor,
        eps: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        theta0: Optional[Tensor] = None,
        t_min: float = 1e-3,
        space: str = "velocity",
        weight: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
        """Train the correction term on joint samples (theta1, eps, y).

        space = "velocity":  E || v_prior(theta_t) + c(t) g - u_t ||^2        (stable, default)
        space = "score"   :  E || g - (u_t - v_prior(theta_t)) / c(t) ||^2     (uniform weight in score space;
                             high variance near t = 0, so keep ``t_min`` away from 0)
        ``weight(t)`` optionally re-weights the per-sample loss. The prior is frozen (no gradient).
        """
        if theta0 is None:
            theta0 = torch.randn_like(theta1)
        if t is None:
            t = torch.rand(theta1.shape[0], device=theta1.device, dtype=theta1.dtype) * (1.0 - t_min) + t_min
        theta_t, u_t = self.path.interpolate(theta1, theta0, t)
        with torch.no_grad():
            v_prior = prior(theta_t, t)
        residual = u_t - v_prior
        c = self.path.score_to_velocity_scale(t, self.t_clip)[:, None]
        g = self(theta_t, t, y, eps, mask)

        if space == "velocity":
            per = ((c * g - residual) ** 2).sum(-1)
        elif space == "score":
            per = ((g - residual / c) ** 2).sum(-1)
        else:
            raise ValueError(f"unknown loss space {space!r}")
        if weight is not None:
            per = per * weight(t)
        return per.mean()


__all__: Sequence[str] = ["timestep_embedding", "MLP", "LinearPath", "PriorFlowModel", "LikelihoodGuidanceModel"]
