"""JADAI-style location finding rollout figures (their Figures 2 and 5) for our flow model.

For every design source (random designs from the prior and each given policy) and the same true sources theta,
runs one rollout of max(t) experiments and shows posterior samples q(theta | h_t), the designs and the ground truth:
    Figure 2: t in {1, 3, 5, 7, 10} plus a corner plot of the posterior at t = 10
    Figure 5: 5 examples (rows) x t in {1, 3, 5, 7, 10, 15, 20, 30}
With the default seeds the true sources equal those of the JADAI figures made with jadai_lf/scripts.

    python scripts/lf_rollout_figures.py --flow-ckpt checkpoints/lf_flow.pt \
        --policy divergence=checkpoints/lf_policy_div.pt --policy density=checkpoints/lf_policy_density.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from train_lf_flow import build_flow
from train_lf_policy import load_policy

from flow_bed.models import PosteriorSampler
from flow_bed.simulators.location_finding import LocationFinding

FIG2_T = [1, 3, 5, 7, 10]
FIG5_T = [1, 3, 5, 7, 10, 15, 20, 30]
EXTENT = (-3, 3)
POST_CMAP = LinearSegmentedColormap.from_list("post", ["#ffffff", "#f7a8d8", "#e7298a", "#b5006b"])
DESIGN_COLOR = "#3fa9f5"


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


def plot_panel(ax, post, designs, theta, t):
    pts = post.reshape(-1, 2)  # both sources share the 2D plane
    ax.hexbin(pts[:, 0], pts[:, 1], gridsize=40, extent=EXTENT * 2, cmap=POST_CMAP, mincnt=1, bins="log", linewidths=0)
    ax.scatter(designs[:t, 0], designs[:t, 1], s=12, color=DESIGN_COLOR, zorder=3)
    ax.scatter(theta[:, 0], theta[:, 1], marker="x", s=60, color="black", linewidths=1.8, zorder=4)
    ax.set_xlim(EXTENT)
    ax.set_ylim(EXTENT)
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend(fig, y):
    handles = [
        Line2D([], [], marker="h", ls="", color="#e7298a", markersize=9, label=r"Posterior $q(\theta \mid h_t)$"),
        Line2D([], [], marker="o", ls="", color=DESIGN_COLOR, markersize=7, label=r"Designs $\xi_t$"),
        Line2D([], [], marker="x", ls="", color="black", markersize=8, mew=2, label=r"Ground Truth $\theta$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, y))


def corner(fig, spec, post, theta):
    x = post.reshape(len(post), -1)  # (N, 4): theta11, theta12, theta21, theta22
    truth = theta.reshape(-1)
    names = [r"$\theta_{11}$", r"$\theta_{12}$", r"$\theta_{21}$", r"$\theta_{22}$"]
    grid = spec.subgridspec(4, 4, hspace=0.08, wspace=0.08)
    for i in range(4):
        for j in range(i + 1):
            ax = fig.add_subplot(grid[i, j])
            if i == j:
                ax.hist(x[:, i], bins=60, range=EXTENT, histtype="step", color="#e7298a")
                ax.axvline(truth[i], color="black", lw=0.6)
                ax.set_yticks([])
            else:
                ax.hexbin(x[:, j], x[:, i], gridsize=30, extent=EXTENT * 2, cmap=POST_CMAP, mincnt=1, bins="log",
                          linewidths=0)
                ax.axvline(truth[j], color="black", lw=0.6)
                ax.axhline(truth[i], color="black", lw=0.6)
                ax.set_ylim(EXTENT)
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=8)
            ax.set_xlim(EXTENT)
            ax.set_xticks([])
            if i == 3:
                ax.set_xlabel(names[j], fontsize=8)


def figure2(rollout, theta, seed, name, out):
    designs, posts = rollout(theta, FIG2_T, seed)
    th = theta.reshape(-1, 2).cpu().numpy()
    fig = plt.figure(figsize=(7.7, 5.2))
    gs = fig.add_gridspec(2, 3, hspace=0.12, wspace=0.12, bottom=0.12)
    for idx, t in enumerate(FIG2_T):
        r, c = divmod(idx, 3)
        ax = fig.add_subplot(gs[r, c])
        plot_panel(ax, posts[t], designs, th, t)
        ax.text(0.04, 0.96, f"$t={t}$", transform=ax.transAxes, va="top", fontsize=12)
        if r == 0:
            ax.set_xticklabels([])
        if c > 0:
            ax.set_yticklabels([])
    corner(fig, gs[1, 2], posts[10], th)
    fig.suptitle(f"Designs: {name}", y=0.98)
    legend(fig, 0.0)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure5(rollout, thetas, seeds, name, out):
    fig, axes = plt.subplots(len(thetas), len(FIG5_T), figsize=(15, 2.0 * len(thetas) + 0.6), sharex=True, sharey=True)
    for r, (theta, seed) in enumerate(zip(thetas, seeds)):
        designs, posts = rollout(theta, FIG5_T, seed)
        th = theta.reshape(-1, 2).cpu().numpy()
        for c, t in enumerate(FIG5_T):
            plot_panel(axes[r, c], posts[t], designs, th, t)
            if r == 0:
                axes[r, c].set_title(f"$t={t}$", fontsize=13)
    fig.suptitle(f"Designs: {name}", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    legend(fig, -0.02)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt", type=Path, default=Path("checkpoints/lf_flow.pt"))
    p.add_argument("--policy", action="append", default=[], help="NAME=PATH of a policy checkpoint (repeatable)")
    p.add_argument("--out", type=Path, default=Path("checkpoints/figures"))
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fig5-rows", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    sim = LocationFinding()
    ck = torch.load(args.flow_ckpt, map_location=dev)
    prior, guidance = build_flow(ck["config"], sim)
    prior.load_state_dict(ck["prior"])
    guidance.load_state_dict(ck["guidance"])
    guidance.to(dev).eval()
    sampler = PosteriorSampler(prior, guidance, solver="dopri5")

    sources = {"random": None}
    for spec in args.policy:
        name, path = spec.split("=", 1)
        pk = torch.load(path, map_location=dev)
        policy = load_policy(pk, guidance, sim).to(dev).eval()
        sources[name] = policy

    # true sources: fixed per example and shared by all design sources
    seeds = [args.seed + 1000 * i for i in range(1 + args.fig5_rows)]
    thetas = []
    for s in seeds:
        torch.manual_seed(s)
        thetas.append(sim.sample_prior(1, dev))  # (1, K * dx)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, policy in sources.items():
        def rollout(theta, ts, seed, policy=policy):
            return rollout_with_posteriors(sim, guidance, sampler, policy, theta, ts, args.n_samples, seed)

        figure2(rollout, thetas[0], seeds[0] + 1, name, args.out / f"fig2_{name}.png")
        figure5(rollout, thetas[1:], [s + 1 for s in seeds[1:]], name, args.out / f"fig5_{name}.png")
        print(f"saved {name} figures to {args.out}", flush=True)


if __name__ == "__main__":
    main()
