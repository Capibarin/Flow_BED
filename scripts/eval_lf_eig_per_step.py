"""True one-step EIG of every design of a location-finding policy on the Figure 5 rollouts, next to the best one-step
design on the same history (true likelihood and true posterior only, so it does not depend on the posterior model).

For each Figure 5 row (same true sources and rollouts as lf_rollout_figures.py) and each t, with samples of the true
p(theta | h_{1:t-1}) (lf_true_posterior.true_posterior_samples; the prior for t = 1):
  policy      EIG(xi_t | h_{1:t-1}) of the policy's design (nested MC, lf_true_posterior.one_step_eig)
  best        max over designs: coarse grid on [-3, 3]^2, Adam ascent (common random numbers) from the 8 best grid
              points and from xi_t, end points evaluated like xi_t
  ascent      the end point of the ascent started at xi_t (is a good design within reach of the policy's?)
The sum of the best one-step EIGs is a reference on the policy's own histories, not a bound on the total EIG. Prints a
per-row table and averages over phases of the rollout; saves eig_per_step_<name>.pt and .png to --out.
Default settings take ~80 min for 5 rows x 30 steps on a V100; --light ~4x less.

    python scripts/eval_lf_eig_per_step.py --flow-ckpt checkpoints/lf_flow.pt \
        --policy divergence=checkpoints/lf_policy_div.pt --out checkpoints/figures_v2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lf_rollout_figures as lrf
import matplotlib.pyplot as plt
import numpy as np
import torch
from lf_true_posterior import one_step_eig, true_posterior_samples
from train_lf_flow import build_flow
from train_lf_policy import load_policy

from flow_bed.simulators.location_finding import LocationFinding

# sample sizes: posterior samples, policy / end-point evaluation (outer, inner), grid (spacing, outer, inner),
# ascent (outer, inner, iterations)
SETTINGS = {
    "full": {"post": 20_000, "eval": (4000, 20_000), "grid": (0.1, 500, 10_000), "ascent": (500, 5000, 150)},
    "light": {"post": 10_000, "eval": (2000, 10_000), "grid": (0.2, 300, 5000), "ascent": (300, 3000, 100)},
}
PHASES = [(1, 1), (2, 4), (5, 10), (11, 20), (21, 30)]


def eig_no_grad(sim, xi, th_in, th_out, noise, chunk):
    with torch.no_grad():
        return torch.cat([one_step_eig(sim, xi[i:i + chunk], th_in, th_out, noise) for i in range(0, len(xi), chunk)])


def step_eigs(sim, post, xi_t, cfg, gen, grid):
    """policy / best / ascent-from-xi_t one-step EIGs for posterior samples post (N, K * dx), policy design xi_t (dx,)"""
    th = post.float()

    def sets(n_out, n_in):
        perm = torch.randperm(th.shape[0], generator=gen, device=th.device)
        return th[perm[:n_in]], th[perm[:n_out]], torch.randn(n_out, device=th.device, generator=gen)

    ev = sets(*cfg["eval"])
    e_pol = eig_no_grad(sim, xi_t[None], *ev, 1).item()
    e_grid = eig_no_grad(sim, grid, *sets(*cfg["grid"][1:]), 16)
    x = torch.cat([grid[e_grid.topk(8).indices], xi_t[None]]).clone().requires_grad_(True)
    n_out, n_in, iters = cfg["ascent"]
    a_in, a_out, a_nz = sets(n_out, n_in)
    opt = torch.optim.Adam([x], lr=0.02)
    for it in range(iters):
        if it == 2 * iters // 3:
            opt.param_groups[0]["lr"] = 0.005
        opt.zero_grad()
        (-one_step_eig(sim, x, a_in, a_out, a_nz).sum()).backward()
        opt.step()
    ends = x.detach()
    e_ends = eig_no_grad(sim, ends, *ev, 1)
    b = int(e_ends.argmax())
    return {"eig_policy": e_pol, "eig_best": e_ends[b].item(), "xi_best": ends[b].cpu().numpy(),
            "eig_ascent": e_ends[-1].item(), "xi_ascent": ends[-1].cpu().numpy()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt", type=Path, default=Path("checkpoints/lf_flow.pt"))
    p.add_argument("--policy", action="append", default=[],
                   help="NAME=PATH of a policy checkpoint (repeatable); NAME=random for random designs")
    p.add_argument("--out", type=Path, default=Path("checkpoints/figures"))
    p.add_argument("--T", type=int, default=30, help="steps per rollout")
    p.add_argument("--rows", type=int, default=5, help="number of Figure 5 rows (rollouts)")
    p.add_argument("--light", action="store_true", help="fewer samples and a coarser grid (~4x faster)")
    p.add_argument("--seed", type=int, default=0, help="as --seed of lf_rollout_figures.py")
    p.add_argument("--sampler-seed", type=int, default=11)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    sim = LocationFinding()
    cfg = SETTINGS["light" if args.light else "full"]
    guidance = None
    if any(spec.split("=", 1)[1] != "random" for spec in args.policy):
        ck = torch.load(args.flow_ckpt, map_location=dev)
        _, guidance = build_flow(ck["config"], sim)
        guidance.load_state_dict(ck["guidance"])
        guidance.to(dev).eval()
    g1 = torch.arange(-3, 3 + 1e-6, cfg["grid"][0], device=dev)
    grid = torch.stack(torch.meshgrid(g1, g1, indexing="xy"), -1).reshape(-1, 2)
    thetas, seeds = lrf.fig5_examples(sim, args.rows, args.seed, dev)
    args.out.mkdir(parents=True, exist_ok=True)

    for spec in args.policy:
        name, path = spec.split("=", 1)
        policy = None if path == "random" else load_policy(torch.load(path, map_location=dev), guidance, sim).to(dev).eval()
        gen = torch.Generator(device=dev).manual_seed(args.sampler_seed)
        res, t0 = [], time.time()
        for r, (theta, seed) in enumerate(zip(thetas, seeds)):
            eps, y = lrf.rollout(sim, policy, theta, args.T, seed)
            recs = []
            for t in range(1, args.T + 1):
                post, info = true_posterior_samples(sim, y[0, :t - 1, 0], eps[0, :t - 1], cfg["post"], gen)
                xi_t = eps[0, t - 1]
                rec = {"t": t, "xi": xi_t.cpu().numpy(), **info, **step_eigs(sim, post, xi_t, cfg, gen, grid)}
                recs.append(rec)
                print(f"{name} row {r + 1} t={t:2d} [{info['modes']} modes, ESS {info['ess']:.3f}]  EIG policy "
                      f"{rec['eig_policy']:6.3f}  best {rec['eig_best']:6.3f} at {rec['xi_best'].round(2)}  ascent "
                      f"from xi_t {rec['eig_ascent']:6.3f}  xi_t {rec['xi'].round(2)}  [{time.time() - t0:.0f}s]",
                      flush=True)
            res.append(recs)
            torch.save({"name": name, "rows": res, "settings": cfg}, args.out / f"eig_per_step_{name}.pt")

        P = np.array([[q["eig_policy"] for q in recs] for recs in res])
        B = np.array([[q["eig_best"] for q in recs] for recs in res])
        A = np.array([[q["eig_ascent"] for q in recs] for recs in res])
        print(f"\n{name}: sum over t of the one-step EIG, per row: policy {P.sum(1).round(2)}, best {B.sum(1).round(2)}")
        for lo, hi in PHASES:
            if lo > args.T:
                break
            s = slice(lo - 1, min(hi, args.T))
            print(f"  t {lo:2d}-{min(hi, args.T):2d}: policy {P[:, s].mean():.2f}  best {B[:, s].mean():.2f}  "
                  f"policy/best {P[:, s].mean() / B[:, s].mean():.2f}  ascent from xi_t {A[:, s].mean():.2f}  "
                  f"steps with policy < 0.2: {(P[:, s] < 0.2).mean():.0%}")

        fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.8), constrained_layout=True, squeeze=False)
        ts = np.arange(1, args.T + 1)
        for r, ax in enumerate(axes[0]):
            ax.plot(ts, B[r], "o-", color="k", ms=3, label="best one-step design")
            ax.plot(ts, P[r], "o-", color=lrf.DESIGN_COLOR, ms=3, label=r"policy design $\xi_t$")
            ax.plot(ts, A[r], "--", color="orange", lw=1, label=r"local ascent from $\xi_t$")
            ax.set_title(f"row {r + 1}: sum {P[r].sum():.1f} / {B[r].sum():.1f}")
            ax.set_xlabel("t")
            ax.set_ylim(bottom=0)
        axes[0, 0].set_ylabel(r"true one-step EIG$(\xi \mid h_{1:t-1})$ [nats]")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{name}: one-step EIG of the policy's designs vs the best design (Figure 5 rollouts)")
        out = args.out / f"eig_per_step_{name}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
