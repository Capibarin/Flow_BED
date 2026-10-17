"""Location finding (as in DAD / JADAI): K signal sources in the plane, measured at design locations.

theta = (theta_1, ..., theta_K), theta_k in R^dx,  theta ~ N(0, I)       (flattened to (B, K * dx) for the flow)
mu(theta, eps) = b + sum_k a / (m + ||eps - theta_k||^2)
y | theta, eps ~ N(log mu(theta, eps), sigma^2)                          (reparametrised, differentiable in eps)
Random designs are drawn from the same prior, eps ~ N(0, I).
Defaults match JADAI: K = 2, dx = 2, a = 1, b = 0.1, m = 1e-4, sigma = 0.5.
"""

from __future__ import annotations

import torch
from torch import Tensor


class LocationFinding:
    def __init__(
        self, n_sources: int = 2, dx: int = 2, a: float = 1.0, b: float = 0.1, m: float = 1e-4, sigma: float = 0.5
    ):
        self.n_sources, self.dx = n_sources, dx
        self.dim = n_sources * dx  # theta dimension seen by the flow
        self.eps_dim, self.y_dim = dx, 1
        self.a, self.b, self.m, self.sigma = a, b, m, sigma

    def sample_prior(self, n: int, device=None) -> Tensor:
        """theta ~ N(0, I), shape (n, K * dx)."""
        return torch.randn(n, self.dim, device=device)

    def sample_design(self, n: int, n_obs: int = 1, device=None) -> Tensor:
        """Random designs eps ~ N(0, I), shape (n, n_obs, dx)."""
        return torch.randn(n, n_obs, self.dx, device=device)

    def log_mu(self, theta: Tensor, eps: Tensor) -> Tensor:
        """theta (B, K * dx), eps (B, n, dx) -> noise-free log signal (B, n, 1)."""
        sources = theta.reshape(-1, 1, self.n_sources, self.dx)  # (B, 1, K, dx)
        dist2 = ((eps[:, :, None, :] - sources) ** 2).sum(-1)  # (B, n, K)
        return torch.log(self.b + (self.a / (self.m + dist2)).sum(-1, keepdim=True))

    def simulate(self, theta: Tensor, eps: Tensor) -> Tensor:
        """theta (B, K * dx), eps (B, n, dx) -> y (B, n, 1)."""
        log_mu = self.log_mu(theta, eps)
        return log_mu + self.sigma * torch.randn_like(log_mu)

    def sample_joint(self, n: int, n_obs: int = 1, device=None) -> tuple[Tensor, Tensor, Tensor]:
        theta = self.sample_prior(n, device)
        eps = self.sample_design(n, n_obs, device)
        return theta, eps, self.simulate(theta, eps)
