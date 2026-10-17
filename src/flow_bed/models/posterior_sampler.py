"""Guided posterior sampling and log-density evaluation with torchdyn.

``VectorField``      dx/dt = v_prior(x, t) + scale * w(x, t, history); with ``divergence`` set, the state
                     is augmented with the log-density: d/dt [x, logp] = [v, -div v].
``PosteriorSampler`` ``sample`` integrates t: 0 -> 1, ``log_prob`` integrates t: 1 -> 0.

``history`` is the output of the guidance model's ``HistoryEncoder``, encoder(y, eps), shape (B, out_dim).
With ``guidance=None`` (or ``history=None``) everything reduces to the learned prior p(theta).
"""

from __future__ import annotations

from math import log, pi

import torch
from torch import Tensor, nn
from torchdyn.core import NeuralODE

from flow_bed.models.flow_matching import LikelihoodGuidanceModel, PriorFlowModel

ADAPTIVE_SOLVERS = ("dopri5", "tsit5")


def exact_divergence(v: Tensor, x: Tensor, create_graph: bool) -> Tensor:
    """Exact trace of dv/dx via d autograd calls. ``v``, ``x``: (B, d) -> (B,)."""
    div = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for i in range(x.shape[1]):
        (grad,) = torch.autograd.grad(v[:, i].sum(), x, create_graph=create_graph, retain_graph=True)
        div = div + grad[:, i]
    return div


def hutchinson_divergence(v: Tensor, x: Tensor, noise: Tensor, create_graph: bool) -> Tensor:
    """Hutchinson estimator e^T (dv/dx) e with a single probe ``noise`` (B, d) -> (B,)."""
    (vjp,) = torch.autograd.grad(v, x, grad_outputs=noise, create_graph=create_graph, retain_graph=True)
    return (vjp * noise).sum(-1)


class VectorField(nn.Module):
    """Posterior velocity for torchdyn, ``forward(t, x)``. Set ``history`` (and ``noise``) before integrating."""

    def __init__(
        self,
        prior: PriorFlowModel,
        guidance: LikelihoodGuidanceModel | None = None,
        guidance_scale: float = 1.0,
        divergence: str | None = None,
    ):
        super().__init__()
        self.prior = prior
        self.guidance = guidance
        self.guidance_scale = guidance_scale
        self.divergence = divergence  # None, "exact" or "hutchinson"
        self.history: Tensor | None = None
        self.noise: Tensor | None = None  # Hutchinson probe, fixed for the whole trajectory

    def velocity(self, t: Tensor, x: Tensor) -> Tensor:
        t = t.expand(x.shape[0])
        v, _ = self.prior(x, t)
        if self.guidance is not None and self.history is not None:
            v = v + self.guidance_scale * self.guidance(x, t, self.history)
        return v

    def forward(self, t: Tensor, x: Tensor, args=None) -> Tensor:
        if self.divergence is None:
            return self.velocity(t, x)

        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            theta = x[:, :-1]
            if not theta.requires_grad:
                theta = theta.detach().requires_grad_(True)
            v = self.velocity(t, theta)
            if self.divergence == "exact":
                div = exact_divergence(v, theta, create_graph)
            else:
                div = hutchinson_divergence(v, theta, self.noise, create_graph)
        if not create_graph:
            v, div = v.detach(), div.detach()
        return torch.cat([v, -div[:, None]], dim=-1)


class PosteriorSampler(nn.Module):
    """Sample theta ~ q(theta | history) and evaluate log q(theta | history).

    solver     : torchdyn solver ('dopri5', 'tsit5', 'rk4', 'euler', ...)
    n_steps    : number of steps for fixed-step solvers (ignored by adaptive ones)
    divergence : 'exact' (fine for low-dim theta) or 'hutchinson'
    """

    def __init__(
        self,
        prior: PriorFlowModel,
        guidance: LikelihoodGuidanceModel | None = None,
        solver: str = "dopri5",
        n_steps: int = 100,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        divergence: str = "exact",
        guidance_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = prior.dim
        self.n_steps = n_steps
        self.adaptive = solver in ADAPTIVE_SOLVERS
        self.vf = VectorField(prior, guidance, guidance_scale)
        self.cnf = VectorField(prior, guidance, guidance_scale, divergence)
        self.node = NeuralODE(self.vf, solver=solver, atol=atol, rtol=rtol)
        self.cnf_node = NeuralODE(self.cnf, solver=solver, atol=atol, rtol=rtol)

    def _t_span(self, t0: float, t1: float, ref: Tensor) -> Tensor:
        n = 2 if self.adaptive else self.n_steps + 1
        return torch.linspace(t0, t1, n, device=ref.device, dtype=ref.dtype)

    def _set_history(self, history: Tensor | None) -> None:
        self.vf.history = history
        self.cnf.history = history

    @staticmethod
    def _log_normal(x: Tensor) -> Tensor:
        return -0.5 * (x**2).sum(-1) - 0.5 * x.shape[-1] * log(2 * pi)

    def sample(self, n_samples: int, history: Tensor | None = None, return_log_prob: bool = False):
        """Draw samples for B encoded histories ``history`` (B, out_dim).

        Returns theta (B, n_samples, d), or (n_samples, d) without history; with ``return_log_prob`` also
        log q(theta | history) of matching leading shape.
        """
        ref = next(self.vf.prior.parameters())
        B = 1 if history is None else history.shape[0]
        N = B * n_samples
        if history is not None:
            history = history.repeat_interleave(n_samples, 0)
        self._set_history(history)
        theta0 = torch.randn(N, self.dim, device=ref.device, dtype=ref.dtype)
        shape = (n_samples,) if history is None else (B, n_samples)

        if not return_log_prob:
            _, sol = self.node(theta0, self._t_span(0.0, 1.0, theta0))
            return sol[-1].reshape(*shape, self.dim)

        self.cnf.noise = torch.randn_like(theta0)
        state = torch.cat([theta0, torch.zeros(N, 1, device=ref.device, dtype=ref.dtype)], dim=-1)
        _, sol = self.cnf_node(state, self._t_span(0.0, 1.0, theta0))
        theta, delta_logp = sol[-1, :, :-1], sol[-1, :, -1]
        logq = self._log_normal(theta0) + delta_logp
        return theta.reshape(*shape, self.dim), logq.reshape(shape)

    def log_prob(self, theta: Tensor, history: Tensor | None = None) -> Tensor:
        """log q(theta | history) for theta (B, d) with one encoded history row per theta, (B, out_dim).

        Integrates the flow backwards 1 -> 0: log q(theta) = log N(theta_0) - int_1^0 div v dt.
        """
        self._set_history(history)
        self.cnf.noise = torch.randn_like(theta)
        state = torch.cat([theta, torch.zeros(theta.shape[0], 1, device=theta.device, dtype=theta.dtype)], dim=-1)
        _, sol = self.cnf_node(state, self._t_span(1.0, 0.0, theta))
        theta0, delta_logp = sol[-1, :, :-1], sol[-1, :, -1]
        return self._log_normal(theta0) - delta_logp
