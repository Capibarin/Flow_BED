"""True posterior and true one-step EIG for location finding (no posterior model), used by the evaluation scripts
lf_true_posterior_figures.py and eval_lf_eig_per_step.py.

true_posterior_samples   samples of p(theta | h) in float64:
                         1. modes of log p~ (Adam from 4096 prior starts + LBFGS refinement)
                         2. sampling-importance-resampling from 2^20 proposals (0.1 prior + 0.9 Laplace Gaussians at the
                            modes, covariance x4)
                         3. 500 MALA steps per point (step size adapted per chain during the first 250) to remove
                            duplicates and the proposal's bias
one_step_eig             EIG(xi | h) = E_{theta ~ p(.|h), y}[log p(y | theta, xi) - log p(y | h, xi)] by nested Monte
                         Carlo from posterior samples; differentiable in xi
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from flow_bed.simulators.location_finding import LocationFinding


def log_mu(sim: LocationFinding, theta: Tensor, eps: Tensor) -> Tensor:
    """theta (..., K * dx), eps (n, dx) -> noise-free log signal (..., n)."""
    src = theta.reshape(*theta.shape[:-1], 1, sim.n_sources, sim.dx)
    d2 = ((eps[:, None, :] - src) ** 2).sum(-1)
    return torch.log(sim.b + (sim.a / (sim.m + d2)).sum(-1))


def log_lik(sim: LocationFinding, theta: Tensor, y: Tensor, eps: Tensor) -> Tensor:
    """log p(h | theta) for y (n,), eps (n, dx) -> (...,)"""
    r = (y - log_mu(sim, theta, eps)) / sim.sigma
    return (-0.5 * r**2 - math.log(sim.sigma) - 0.5 * math.log(2 * math.pi)).sum(-1)


def log_post(sim: LocationFinding, theta: Tensor, y: Tensor, eps: Tensor) -> Tensor:
    """unnormalised log p(theta | h), the N(0, I) prior without its constant"""
    return -0.5 * (theta**2).sum(-1) + log_lik(sim, theta, y, eps)


def _grad_log_post(sim, x, y, eps):
    x = x.detach().requires_grad_(True)
    lp = log_post(sim, x, y, eps)
    (g,) = torch.autograd.grad(lp.sum(), x)
    return lp.detach(), g


def find_modes(sim: LocationFinding, y: Tensor, eps: Tensor, gen: torch.Generator, max_modes: int = 32) -> list[Tensor]:
    x = torch.randn(4096, sim.dim, device=y.device, dtype=torch.float64, generator=gen).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)
    for i in range(1500):
        if i == 1000:
            opt.param_groups[0]["lr"] = 0.005
        opt.zero_grad()
        (-log_post(sim, x, y, eps).sum()).backward()
        opt.step()
    with torch.no_grad():
        lp = log_post(sim, x, y, eps)
    modes = []
    for c in x.detach()[lp.argsort(descending=True)[:256]]:
        if len(modes) >= max_modes or any((c - m).norm() < 1e-3 for m in modes):
            continue
        z = c.clone().requires_grad_(True)
        lb = torch.optim.LBFGS([z], lr=1, max_iter=200, tolerance_grad=1e-12, tolerance_change=1e-14,
                               line_search_fn="strong_wolfe")

        def closure(z=z, lb=lb):
            lb.zero_grad()
            loss = -log_post(sim, z, y, eps)
            loss.backward()
            return loss
        lb.step(closure)
        z = z.detach()
        if not any((z - m).norm() < 1e-4 for m in modes):
            modes.append(z)
    return modes


def true_posterior_samples(sim: LocationFinding, y: Tensor, eps: Tensor, n_samples: int, gen: torch.Generator,
                           n_proposals: int = 2**20) -> tuple[Tensor, dict]:
    """Samples (n_samples, K * dx) of p(theta | h) for one history y (n,), eps (n, dx), in float64 (n = 0: prior).

    Returns the samples and diagnostics {"modes", "ess" (IS efficiency of the SIR stage), "acc" (MALA acceptance)}.
    """
    dev = y.device
    y, eps = y.double(), eps.double()
    if y.shape[0] == 0:
        x = torch.randn(n_samples, sim.dim, device=dev, dtype=torch.float64, generator=gen)
        return x, {"modes": 1, "ess": 1.0, "acc": 1.0}
    modes = find_modes(sim, y, eps, gen)
    eye = torch.eye(sim.dim, device=dev, dtype=torch.float64)
    comps = []
    for m in modes:
        H = torch.autograd.functional.hessian(lambda t: -log_post(sim, t, y, eps), m)
        ev, V = torch.linalg.eigh(0.5 * (H + H.T))
        cov = (V / ev.clamp_min(1e-6)) @ V.T * 4.0 + 1e-12 * eye
        comps.append(torch.distributions.MultivariateNormal(m, covariance_matrix=cov))
    dists = [torch.distributions.MultivariateNormal(torch.zeros(sim.dim, device=dev, dtype=torch.float64), eye)] + comps
    w_mix = torch.tensor([0.1] + [0.9 / len(comps)] * len(comps), device=dev, dtype=torch.float64)
    idx = torch.multinomial(w_mix, n_proposals, replacement=True, generator=gen)
    xs = torch.cat([dists[j].sample((int((idx == j).sum()),)) for j in range(len(dists))])
    log_r = torch.logsumexp(torch.stack([d.log_prob(xs) for d in dists]) + w_mix.log()[:, None], 0)
    logw = torch.cat([log_post(sim, xs[i:i + 2**16], y, eps) for i in range(0, n_proposals, 2**16)]) - log_r
    w = torch.softmax(logw, 0)
    ess = (1 / (w**2).sum() / n_proposals).item()
    x = xs[torch.multinomial(w, n_samples, replacement=True, generator=gen)]

    # MALA rejuvenation, per-chain step size
    step = torch.full((n_samples, 1), 1e-3, device=dev, dtype=torch.float64)
    lp, g = _grad_log_post(sim, x, y, eps)
    acc_sum = torch.zeros(n_samples, device=dev, dtype=torch.float64)
    for it in range(500):
        noise = torch.randn(x.shape, device=dev, dtype=torch.float64, generator=gen)
        prop = x + step * g + (2 * step).sqrt() * noise
        lp_p, g_p = _grad_log_post(sim, prop, y, eps)
        log_q_fwd = -((prop - x - step * g) ** 2).sum(-1) / (4 * step[:, 0])
        log_q_bwd = -((x - prop - step * g_p) ** 2).sum(-1) / (4 * step[:, 0])
        log_a = (lp_p - lp + log_q_bwd - log_q_fwd).clamp(max=0)
        accept = torch.rand(n_samples, device=dev, dtype=torch.float64, generator=gen).log() < log_a
        x = torch.where(accept[:, None], prop, x)
        lp = torch.where(accept, lp_p, lp)
        g = torch.where(accept[:, None], g_p, g)
        if it < 250:  # adapt towards ~0.57 acceptance, then keep fixed (valid MCMC afterwards)
            step = step * torch.exp(0.1 * (log_a.exp()[:, None] - 0.57))
        else:
            acc_sum += accept.double()
    return x, {"modes": len(modes), "ess": ess, "acc": (acc_sum / 250).mean().item()}


def one_step_eig(sim: LocationFinding, xi: Tensor, th_in: Tensor, th_out: Tensor, noise: Tensor) -> Tensor:
    """Nested-MC one-step EIG at designs xi (g, dx), given posterior samples th_in (N, K * dx) for the marginal and
    th_out (M, K * dx) with standard normal noise (M,) for the outer expectation -> (g,). Differentiable in xi; with
    fixed th_in, th_out, noise (common random numbers) it is a smooth function of xi."""
    mu_out = log_mu(sim, th_out, xi).T  # (g, M)
    y = mu_out + sim.sigma * noise[None]
    mu_in = log_mu(sim, th_in, xi).T  # (g, N)
    lp = -0.5 * ((y[:, :, None] - mu_in[:, None, :]) / sim.sigma) ** 2
    log_marg = torch.logsumexp(lp, 2) - math.log(th_in.shape[0])
    return (-0.5 * noise[None] ** 2 - log_marg).mean(1)
