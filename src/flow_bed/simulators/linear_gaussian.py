"""Toy linear-Gaussian BED problem with an analytic posterior (for sanity checks).

theta ~ N(0, I_d),   y | theta, eps ~ N(eps^T theta, noise^2),   eps in R^d is the design.
"""

from __future__ import annotations

import torch
from torch import Tensor


class LinearGaussian:
    def __init__(self, dim: int = 2, noise: float = 0.5):
        self.dim, self.noise = dim, noise

    # ---- sampling -----------------------------------------------------------------------
    def sample_prior(self, n: int, device=None) -> Tensor:
        return torch.randn(n, self.dim, device=device)

    def sample_design(self, n: int, n_obs: int = 1, device=None) -> Tensor:
        """Random unit-norm designs, shape (n, n_obs, dim)."""
        e = torch.randn(n, n_obs, self.dim, device=device)
        return e / e.norm(dim=-1, keepdim=True)

    def simulate(self, theta: Tensor, eps: Tensor) -> Tensor:
        """theta (B, d), eps (B, n, d) -> y (B, n, 1)."""
        mean = (eps * theta[:, None, :]).sum(-1, keepdim=True)
        return mean + self.noise * torch.randn_like(mean)

    def sample_joint(self, n: int, n_obs: int = 1, device=None) -> tuple[Tensor, Tensor, Tensor]:
        theta = self.sample_prior(n, device)
        eps = self.sample_design(n, n_obs, device)
        return theta, eps, self.simulate(theta, eps)

    # ---- ground truth -------------------------------------------------------------------
    def log_likelihood(self, theta: Tensor, y: Tensor, eps: Tensor) -> Tensor:
        """sum_i log N(y_i | eps_i^T theta, noise^2); theta (B, d), y (B, n, 1), eps (B, n, d) -> (B,)."""
        mean = (eps * theta[:, None, :]).sum(-1, keepdim=True)
        ll = -0.5 * ((y - mean) / self.noise) ** 2 - torch.log(torch.tensor(self.noise)) - 0.5 * torch.log(torch.tensor(2 * torch.pi))
        return ll.sum(dim=(1, 2))

    def posterior(self, y: Tensor, eps: Tensor) -> tuple[Tensor, Tensor]:
        """Analytic Gaussian posterior for one context: y (n, 1), eps (n, d) -> (mean (d,), cov (d, d))."""
        prec = torch.eye(self.dim, device=y.device) + eps.T @ eps / self.noise**2
        cov = torch.linalg.inv(prec)
        mean = cov @ (eps.T @ y[:, 0]) / self.noise**2
        return mean, cov

    def posterior_log_prob(self, theta: Tensor, y: Tensor, eps: Tensor) -> Tensor:
        """log p(theta | y, eps) for theta (B, d), single context y (n, 1), eps (n, d)."""
        mean, cov = self.posterior(y, eps)
        return torch.distributions.MultivariateNormal(mean, cov).log_prob(theta)
