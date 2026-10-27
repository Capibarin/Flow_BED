"""Figure 5 of lf_rollout_figures.py with samples from the TRUE posterior p(theta | h_t) instead of the flow's.

Same rows, true sources, seeds and design sequences as lf_rollout_figures.py's fig5_<name>.png for the same policy (the
rollouts consume the random numbers identically), same plotting code; the posterior samples come from
lf_true_posterior.true_posterior_samples. Separates errors of the posterior model from those of the designs: if
the two figures agree, the designs are to blame. The flow checkpoint is needed only for policies that read its
history encoder.

    python scripts/lf_true_posterior_figures.py --flow-ckpt checkpoints/lf_flow.pt \
        --policy divergence=checkpoints/lf_policy_div.pt --out checkpoints/figures_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lf_rollout_figures as lrf
import torch
from lf_true_posterior import true_posterior_samples
from matplotlib.lines import Line2D
from train_lf_flow import build_flow
from train_lf_policy import load_policy

from flow_bed.simulators.location_finding import LocationFinding


def legend(fig, y):  # as lrf.legend, labelled as the true posterior
    handles = [
        Line2D([], [], marker="h", ls="", color="#e7298a", markersize=9, label=r"True posterior $p(\theta \mid h_t)$"),
        Line2D([], [], marker="o", ls="", color=lrf.DESIGN_COLOR, markersize=7, label=r"Designs $\xi_t$"),
        Line2D([], [], marker="x", ls="", color="black", markersize=8, mew=2, label=r"Ground Truth $\theta$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, y))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt", type=Path, default=Path("checkpoints/lf_flow.pt"))
    p.add_argument("--policy", action="append", default=[],
                   help="NAME=PATH of a policy checkpoint (repeatable); NAME=random for random designs")
    p.add_argument("--out", type=Path, default=Path("checkpoints/figures"))
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0, help="as --seed of lf_rollout_figures.py")
    p.add_argument("--sampler-seed", type=int, default=123)
    p.add_argument("--fig5-rows", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    sim = LocationFinding()
    guidance = None
    if any(spec.split("=", 1)[1] != "random" for spec in args.policy):
        ck = torch.load(args.flow_ckpt, map_location=dev)
        _, guidance = build_flow(ck["config"], sim)
        guidance.load_state_dict(ck["guidance"])
        guidance.to(dev).eval()
    thetas, seeds = lrf.fig5_examples(sim, args.fig5_rows, args.seed, dev)
    gen = torch.Generator(device=dev).manual_seed(args.sampler_seed)
    lrf.legend = legend
    args.out.mkdir(parents=True, exist_ok=True)

    for spec in args.policy:
        name, path = spec.split("=", 1)
        policy = None if path == "random" else load_policy(torch.load(path, map_location=dev), guidance, sim).to(dev).eval()

        def rollout(theta, ts, seed, policy=policy, name=name):
            eps, y = lrf.rollout(sim, policy, theta, max(ts), seed)
            posts = {}
            for t in ts:
                x, info = true_posterior_samples(sim, y[0, :t, 0], eps[0, :t], args.n_samples, gen)
                posts[t] = x.float().reshape(args.n_samples, sim.n_sources, sim.dx).cpu().numpy()
                print(f"{name} seed {seed} t={t:2d}: {info['modes']} modes, IS-ESS {info['ess']:.3f}, "
                      f"MALA acceptance {info['acc']:.2f}", flush=True)
            return eps[0].cpu().numpy(), posts

        out = args.out / f"fig5_true_posterior_{name}.png"
        lrf.figure5(rollout, thetas, seeds, f"{name} (true posterior)", out)
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
