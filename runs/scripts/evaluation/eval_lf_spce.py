"""sPCE lower and sNMC upper bound on the total EIG of location-finding design sequences.

For each run, theta_0 ~ p(theta), T designs from a policy (or random designs from the prior) with observations
y ~ p(y | theta_0, eps), and L contrastive samples theta_1..L ~ p(theta):
    sPCE = log p(h | theta_0) - log( sum_{l=0..L} p(h | theta_l) ) + log(L + 1)    (lower bound, at most log(L + 1))
    sNMC = log p(h | theta_0) - log( sum_{l=1..L} p(h | theta_l) ) + log(L)        (upper bound)
Both use the same rollouts and contrastive samples, and only the true likelihood, so they do not depend on the
posterior model.

    python runs/scripts/evaluation/eval_lf_spce.py --flow-ckpt checkpoints/lf_flow.pt --policy div_exact=checkpoints/lf_policy_div_exact.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import Tensor

from flow_bed.lf.builders import build_flow, load_policy, resolve_sim
from flow_bed.simulators.location_finding import LocationFinding


def log_lik(sim: LocationFinding, theta: Tensor, y: Tensor, eps: Tensor) -> Tensor:
    """theta (R, C, d), y (R, T, 1), eps (R, T, dx) -> log p(y | theta, eps) of shape (R, C)."""
    R, C, _ = theta.shape
    T = y.shape[1]
    eps_rc = eps[:, None].expand(R, C, T, sim.dx).reshape(R * C, T, sim.dx)
    log_mu = sim.log_mu(theta.reshape(R * C, -1), eps_rc).reshape(R, C, T)
    ll = -0.5 * ((y[:, None, :, 0] - log_mu) / sim.sigma) ** 2 - math.log(sim.sigma) - 0.5 * math.log(2 * math.pi)
    return ll.sum(-1)


@torch.no_grad()
def spce(sim, policy, T, L, runs, run_batch, chunk, dev):
    """(mean, std, s.e.) over the runs of sPCE and of sNMC"""
    lower, upper = [], []
    for r0 in range(0, runs, run_batch):
        R = min(run_batch, runs - r0)
        theta0 = sim.sample_prior(R, dev)
        eps = torch.zeros(R, 0, sim.eps_dim, device=dev)
        y = torch.zeros(R, 0, sim.y_dim, device=dev)
        for _ in range(T):
            xi = sim.sample_design(R, 1, dev) if policy is None else policy(y, eps)[:, None]
            eps = torch.cat([eps, xi], dim=1)
            y = torch.cat([y, sim.simulate(theta0, xi)], dim=1)
        ll0 = log_lik(sim, theta0[:, None], y, eps)[:, 0]  # (R,)
        lse = torch.full_like(ll0, -math.inf)  # running logsumexp over the contrastive samples
        for c0 in range(0, L, chunk):
            C = min(chunk, L - c0)
            ll = log_lik(sim, sim.sample_prior(R * C, dev).reshape(R, C, -1), y, eps)
            lse = torch.logaddexp(lse, torch.logsumexp(ll, dim=1))
        lower.append(ll0 - torch.logaddexp(lse, ll0) + math.log(L + 1))
        upper.append(ll0 - lse + math.log(L))
    stats = [(v.mean().item(), v.std().item(), v.std().item() / math.sqrt(runs)) for v in map(torch.cat, (lower, upper))]
    return tuple(stats)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt", type=Path, default=Path("checkpoints/lf_flow.pt"))
    p.add_argument("--policy", action="append", default=[], help="NAME=PATH of a policy checkpoint (repeatable)")
    p.add_argument("--no-random", action="store_true", help="skip the random-design baseline")
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--L", type=int, default=500_000, help="number of contrastive samples")
    p.add_argument("--runs", type=int, default=4096, help="number of evaluation runs")
    p.add_argument("--run-batch", type=int, default=16)
    p.add_argument("--chunk", type=int, default=20_000, help="contrastive samples per likelihood evaluation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=None,
                   help="observation noise std of y for random designs alone (default 0.5); with a flow: the flow's")
    p.add_argument("--m", type=float, default=None,
                   help="offset m in a / (m + |xi - theta_k|^2) for random designs alone (default 1e-4); "
                        "with a flow: the flow's")
    args = p.parse_args()

    dev = torch.device(args.device)
    ck = torch.load(args.flow_ckpt, map_location=dev) if args.policy else None
    try:
        sim = resolve_sim(ck["config"] if ck is not None else None, sigma=args.sigma, m=args.m)
    except ValueError as e:
        p.error(str(e))
    sources = {} if args.no_random else {"random": None}
    if ck is not None:
        _, guidance = build_flow(ck["config"], sim)
        guidance.load_state_dict(ck["guidance"])
        guidance.to(dev).eval()
        for spec in args.policy:
            name, path = spec.split("=", 1)
            pk = torch.load(path, map_location=dev)
            policy = load_policy(pk, guidance, sim).to(dev).eval()
            sources[f"{name} (epoch {pk.get('epoch', '?')})"] = policy

    print(f"sPCE (lower bound) and sNMC (upper bound), T={args.T}, L={args.L}, runs={args.runs}", flush=True)
    for name, policy in sources.items():
        torch.manual_seed(args.seed)
        (m, s, se), (mu, su, seu) = spce(sim, policy, args.T, args.L, args.runs, args.run_batch, args.chunk, dev)
        print(f"  {name:28s} {m:.3f} ± {s:.3f}  (s.e. {se:.3f})   sNMC {mu:.3f} ± {su:.3f}  (s.e. {seu:.3f})",
              flush=True)


if __name__ == "__main__":
    main()
