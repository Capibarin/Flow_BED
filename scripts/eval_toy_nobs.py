"""Posterior quality vs. number of observations on the toy linear-Gaussian problem.

For each n in --n-obs, draws --n-contexts true thetas with n random designs and observations, samples the guided
flow posterior and compares it with the analytic posterior:
    mean error : ||Sigma^{-1/2} (mu_flow - mu)||, the mean error in units of posterior standard deviations
    cov error  : ||Sigma_flow - Sigma||_F / ||Sigma||_F
    corr       : correlation between log q(theta | y, eps) of the flow and the analytic log p(theta | y, eps)

    python scripts/eval_toy_nobs.py --ckpt checkpoints/toy.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flow_bed.models import LikelihoodGuidanceModel, PosteriorSampler, PriorFlowModel
from flow_bed.models.flow_matching import LEGACY_ACTIVATION
from flow_bed.models.history_encoder import HistoryEncoder
from flow_bed.simulators import LinearGaussian


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/toy.pt"))
    p.add_argument("--n-obs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--n-contexts", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--solver", default="dopri5")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=dev)
    a = ck["args"]
    act = a.get("activation", LEGACY_ACTIVATION)
    prior = PriorFlowModel(a["dim"], hidden_dims=a["hidden_dims"], activation=act)
    prior.load_state_dict(ck["prior"])
    encoder = HistoryEncoder(y_dim=1, eps_dim=a["dim"], hidden_dims=a["hidden_dims"], activation=act)
    guidance = LikelihoodGuidanceModel(prior, encoder, hidden_dims=a["hidden_dims"], activation=act)
    guidance.load_state_dict(ck["guidance"])
    guidance.to(dev).eval()
    sampler = PosteriorSampler(prior, guidance, solver=args.solver)
    sim = LinearGaussian(a["dim"], a["noise"])

    print(f"trained with 1..{a['max_obs']} observations per batch; solver {args.solver}")
    print(f"{'n_obs':>5} | {'mean err [std]':>17} | {'cov err (rel.)':>17} | {'corr(logq, logp)':>17}")
    torch.manual_seed(args.seed)
    for n in args.n_obs:
        _, eps, y = sim.sample_joint(args.n_contexts, n, dev)
        with torch.no_grad():
            theta, logq = sampler.sample(args.n_samples, encoder(y, eps), return_log_prob=True)

        mean_err, cov_err, corr = [], [], []
        for b in range(args.n_contexts):
            mu, cov = sim.posterior(y[b], eps[b])
            chol = torch.linalg.cholesky(cov)
            diff = torch.linalg.solve_triangular(chol, (theta[b].mean(0) - mu)[:, None], upper=False)
            mean_err.append(diff.norm())
            cov_err.append(torch.linalg.norm(torch.cov(theta[b].T) - cov) / torch.linalg.norm(cov))
            logp = sim.posterior_log_prob(theta[b], y[b], eps[b])
            corr.append(torch.corrcoef(torch.stack([logq[b], logp]))[0, 1])

        def fmt(xs):
            xs = torch.stack(xs)
            return f"{xs.mean():.3f} ± {xs.std() / len(xs) ** 0.5:.3f}"

        print(f"{n:>5} | {fmt(mean_err):>17} | {fmt(cov_err):>17} | {fmt(corr):>17}")


if __name__ == "__main__":
    main()
