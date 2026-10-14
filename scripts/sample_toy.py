"""Guided posterior sampling + log-likelihood on the toy problem, checked against the analytic posterior.

    python scripts/sample_toy.py --ckpt checkpoints/toy.pt --n-obs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flow_bed.models import LikelihoodGuidanceModel, PosteriorSampler, PriorFlowModel
from flow_bed.simulators import LinearGaussian


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/toy.pt"))
    p.add_argument("--n-obs", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--solver", default="dopri5")
    p.add_argument("--divergence", default="exact", choices=["exact", "hutchinson"])
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    prior = PriorFlowModel(a["dim"], hidden=a["hidden"])
    guidance = LikelihoodGuidanceModel(a["dim"], y_dim=1, eps_dim=a["dim"], hidden=a["hidden"])
    prior.load_state_dict(ck["prior"])
    guidance.load_state_dict(ck["guidance"])
    prior.eval()
    guidance.eval()

    torch.manual_seed(args.seed)
    sim = LinearGaussian(a["dim"], a["noise"])
    theta_true, eps, y = sim.sample_joint(1, args.n_obs)  # one context, n_obs observations

    sampler = PosteriorSampler(prior, guidance, solver=args.solver, divergence=args.divergence)
    with torch.no_grad():
        theta, logq = sampler.sample(args.n_samples, y=y, eps=eps, return_log_prob=True)
    theta, logq = theta[0], logq[0]

    mean, cov = sim.posterior(y[0], eps[0])
    print(f"true theta            : {theta_true[0].numpy()}")
    print(f"analytic post. mean   : {mean.numpy()}")
    print(f"flow     post. mean   : {theta.mean(0).numpy()}")
    print(f"analytic post. cov    :\n{cov.numpy()}")
    print(f"flow     post. cov    :\n{torch.cov(theta.T).numpy()}")

    # log-density consistency: forward (sampling) vs backward (log_prob) and vs analytic
    with torch.no_grad():
        logq_bwd = sampler.log_prob(theta[:200], y=y.expand(200, -1, -1), eps=eps.expand(200, -1, -1))
    logp_true = sim.posterior_log_prob(theta[:200], y[0], eps[0])
    print(f"|logq_fwd - logq_bwd| mean : {(logq[:200] - logq_bwd).abs().mean():.4f}")
    print(f"corr(logq, analytic logp)  : {torch.corrcoef(torch.stack([logq_bwd, logp_true]))[0, 1]:.4f}")


if __name__ == "__main__":
    main()
