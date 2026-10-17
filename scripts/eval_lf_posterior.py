"""Quality of the location-finding flow (prior and posteriors with random designs) against the true posterior.

The true posterior is known up to its normaliser:
    log p~(theta | h) = log N(theta; 0, I) + sum_i log N(y_i; log mu(theta, eps_i), sigma^2)
For each n, n_contexts true thetas with n random designs; per context, S flow samples with log q from the ODE.
    prior      moments of learned prior samples, KL(q||p) on its samples, KL(p||q) on true prior samples
    ESS/N      effective sample size of importance weights p~ / q, normalised (1 = perfect)
    KL(q||p)   E_q[log q - log p~] + log Z, with log Z estimated by importance sampling (0 = perfect)
    corr       correlation of log q and log p~ at the flow samples
    calibration  rank r = P_q(log q(theta) > log q(theta_true)); for a calibrated posterior r ~ U[0, 1], so the true
               theta lies in the 50% / 90% highest-density regions in 50% / 90% of the contexts

    python scripts/eval_lf_posterior.py --flow-ckpt checkpoints/lf_flow.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import Tensor
from train_lf_flow import build_flow

from flow_bed.models import PosteriorSampler
from flow_bed.simulators.location_finding import LocationFinding


def log_normal(x: Tensor) -> Tensor:
    return -0.5 * (x**2).sum(-1) - 0.5 * x.shape[-1] * math.log(2 * math.pi)


def log_post_unnorm(sim: LocationFinding, theta: Tensor, y: Tensor, eps: Tensor) -> Tensor:
    """theta (S, d), y (n, 1), eps (n, dx) -> (S,)."""
    S = theta.shape[0]
    log_mu = sim.log_mu(theta, eps[None].expand(S, -1, -1))  # (S, n, 1)
    ll = -0.5 * ((y[None] - log_mu) / sim.sigma) ** 2 - math.log(sim.sigma) - 0.5 * math.log(2 * math.pi)
    return log_normal(theta) + ll.sum(dim=(1, 2))


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt", type=Path, default=Path("checkpoints/lf_flow.pt"))
    p.add_argument("--n-obs", type=int, nargs="+", default=[1, 5, 10, 20, 30])
    p.add_argument("--n-contexts", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = torch.device(args.device)
    sim = LocationFinding()
    ck = torch.load(args.flow_ckpt, map_location=dev)
    prior, guidance = build_flow(ck["config"], sim)
    prior.load_state_dict(ck["prior"])
    guidance.load_state_dict(ck["guidance"])
    guidance.to(dev).eval()
    sampler = PosteriorSampler(prior, guidance, solver="dopri5", divergence="exact")
    torch.manual_seed(args.seed)

    th, logq = sampler.sample(20_000, return_log_prob=True)
    lp = log_normal(th)
    th_true = sim.sample_prior(20_000, dev)
    logq_at_true = sampler.log_prob(th_true)
    print(f"{args.flow_ckpt}")
    print(f"PRIOR: mean max |.| {th.mean(0).abs().max():.3f}   cov - I max |.| "
          f"{(torch.cov(th.T) - torch.eye(sim.dim, device=dev)).abs().max():.3f}   "
          f"|log q - log N| {(logq - lp).abs().mean():.3f}   corr {torch.corrcoef(torch.stack([logq, lp]))[0, 1]:.4f}")
    print(f"       KL(q||p) {(logq - lp).mean():.4f}   KL(p||q) {(log_normal(th_true) - logq_at_true).mean():.4f}")

    S, C = args.n_samples, args.n_contexts
    print(f"POSTERIOR (random designs, {C} contexts x {S} samples)")
    print(f"{'n':>3} | {'ESS/N':>6} | {'KL(q||p)':>9} | {'corr':>6} | {'rank mean':>9} | {'in 50%':>6} | {'in 90%':>6}")
    for n in args.n_obs:
        theta_true, eps, y = sim.sample_joint(C, n, dev)
        history = guidance.encoder(y, eps)
        samples, logq = sampler.sample(S, history, return_log_prob=True)  # (C, S, d), (C, S)
        logq_true = sampler.log_prob(theta_true, history)  # (C,)
        ess, kl, corr, ranks = [], [], [], []
        for c in range(C):
            lpu = log_post_unnorm(sim, samples[c], y[c], eps[c])
            logw = lpu - logq[c]
            w = torch.softmax(logw, 0)
            ess.append(1.0 / (w**2).sum() / S)
            kl.append((logq[c] - lpu).mean() + torch.logsumexp(logw, 0) - math.log(S))
            corr.append(torch.corrcoef(torch.stack([logq[c], lpu]))[0, 1])
            ranks.append((logq[c] > logq_true[c]).float().mean())
        ess, kl, corr, ranks = (torch.stack(v) for v in (ess, kl, corr, ranks))
        print(f"{n:>3} | {ess.mean():6.3f} | {kl.mean():9.3f} | {corr.mean():6.3f} | {ranks.mean():9.3f} | "
              f"{(ranks < 0.5).float().mean():6.2f} | {(ranks < 0.9).float().mean():6.2f}", flush=True)


if __name__ == "__main__":
    main()
