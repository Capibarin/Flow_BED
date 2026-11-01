"""Single location-finding rollouts and the true sources of the Figure 5 examples."""

from __future__ import annotations

import torch


@torch.no_grad()
def rollout(sim, policy, theta, n, seed):
    """One rollout of n experiments for theta (1, K * dx) (policy=None: random designs) -> eps (1, n, dx), y (1, n, 1)."""
    torch.manual_seed(seed)
    eps = torch.zeros(1, 0, sim.eps_dim, device=theta.device)
    y = torch.zeros(1, 0, sim.y_dim, device=theta.device)
    for _ in range(n):
        xi = sim.sample_design(1, 1, theta.device) if policy is None else policy(y, eps)[:, None]
        eps = torch.cat([eps, xi], dim=1)
        y = torch.cat([y, sim.simulate(theta, xi)], dim=1)
    return eps, y


def fig5_examples(sim, n_rows, seed, dev):
    """True sources theta (1, K * dx) and rollout seeds of the Figure 5 rows (as in main)."""
    thetas, seeds = [], []
    for i in range(1, 1 + n_rows):
        torch.manual_seed(seed + 1000 * i)
        thetas.append(sim.sample_prior(1, dev))
        seeds.append(seed + 1000 * i + 1)
    return thetas, seeds


@torch.no_grad()
def rollout_with_posteriors(sim, guidance, sampler, policy, theta, ts, n_samples, seed):
    """One rollout of max(ts) experiments (policy=None: random designs); posterior samples for every t in ts."""
    eps, y = rollout(sim, policy, theta, max(ts), seed)
    posts = {}
    for t in ts:
        torch.manual_seed(seed + t)
        history = guidance.encoder(y[:, :t], eps[:, :t])
        posts[t] = sampler.sample(n_samples, history)[0].reshape(n_samples, sim.n_sources, sim.dx).cpu().numpy()
    return eps[0].cpu().numpy(), posts
