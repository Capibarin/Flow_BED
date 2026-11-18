"""iDAD Figure 5 for one SIR policy, or for the baselines (POLICY = equal | random):
(a) the three epidemic paths of iDAD's Figure 5a (R = beta / gamma = 1.3, 3.0, 8.0), recovered exactly from the vector
    graphics of the paper's PDF (idad_fig5_paths.json: I on their 10000-point grid, 0..100), with the measurements of
    the designs marked;
(b) the designs tau_t, in iDAD's style (one row per experiment t, t = 1 at the bottom; random designs sorted in time);
(c) the flow posterior p(beta, gamma | h_T) of the R = 3.0 path on a grid; red cross: the true value
    (iDAD's plot_epidemic_posterior.py), dotted line: beta / gamma = 3.0. The window is the same in every figure, so
    the posteriors of different designs compare directly.
theta of the three paths: iDAD's values (0.1977, 0.1521), (0.3332, 0.1103), (0.7399, 0.0924).
Panel (d) (--panel-d kde): the same samples smoothed by a Gaussian KDE in log theta (kde_density).
Panel (c): 2D histogram of N_SAMPLES posterior draws (forward flow ODE, dopri5, tol 1e-5), no smoothing; drawn in
chunks with guarded_sampling's guard against dopri5 hangs.
    fig5_sir.py FLOW POLICY NAME [--out-dir DIR]  -> DIR/fig5_sir_NAME_grid_kde.png (default: (c) exact grid density,
    (d) KDE; --panel-c samples / --panel-d none for the histogram-only version)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

from flow_bed.config import parse_args as parse_config_args

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flow_bed.models import PosteriorSampler
from flow_bed.models.guarded_sampling import guarded_log_prob, guarded_sample
from flow_bed.models.policy import Policy
from flow_bed.simulators import sir
from flow_bed.sir import SPEC, load_flow, rollout_designs

THETAS = [(0.1977, 0.1521), (0.3332, 0.1103), (0.7399, 0.0924)]
R_LABELS = ["1.3", "3.0", "8.0"]
# iDAD's Figure 5 style: colours of the paths, markers of the designs
COLORS = [(0.871, 0.796, 0.894), (0.0, 0.0, 0.0), (0.996, 0.851, 0.651)]
MARKERS = [{"marker": "^", "s": 55, "facecolor": COLORS[0], "edgecolor": "k", "linewidth": 0.6},
           {"marker": (8, 2, 0), "s": 80, "color": "k", "linewidth": 0.8},
           {"marker": "D", "s": 40, "facecolor": COLORS[2], "edgecolor": "k", "linewidth": 0.6}]
OFFSETS = [-0.27, 0.0, 0.27]  # vertical position of R = 1.3, 3.0, 8.0 inside a row of panel (b)
N_BINS = 80  # histogram bins per axis over the window of panel (c)
N_SAMPLES = 20_000
N_GRID = 300  # --panel-c grid: points per axis (as iDAD)
BETA_LIM, GAMMA_LIM = (0.15, 0.75), (0.04, 0.22)  # window of panel (c), fixed for all figures
PATHS = Path(__file__).with_name("idad_fig5_paths.json")


@torch.no_grad()
def kde_density(z: torch.Tensor, beta_lim, gamma_lim, n_grid: int, n_val: int = 2000, chunk: int = 4096):
    """Gaussian KDE of posterior samples z (n, 2) (z = (log theta - LOC) / SCALE, i.e. log theta up to an affine map)
    with kernel covariance f^2 Cov(z); the factor f maximises the held-out log-likelihood of n_val samples (KDE on
    the others) over a grid of multiples of Scott's factor. Returns the density of theta on the cell centres of an
    n_grid^2 grid over the window (normalised over the window; Jacobian d z / d theta = 1 / (SCALE theta)), f, info."""
    n = z.shape[0]
    L = torch.linalg.cholesky(torch.cov(z.T))
    w = torch.linalg.solve_triangular(L, z.T, upper=False).T  # whitened samples
    log_det_L = L.diagonal().log().sum()

    def log_kde(x, data, f):  # x, data whitened; log density in z
        out = []
        for xc in x.split(chunk):
            d2 = torch.cdist(xc, data).pow(2)
            out.append(torch.logsumexp(-0.5 * d2 / f**2, 1) - math.log(data.shape[0]) - math.log(2 * math.pi * f**2))
        return torch.cat(out) - log_det_L

    perm = torch.randperm(n, device=z.device)
    val, train = w[perm[:n_val]], w[perm[n_val:]]
    scott = (n - n_val) ** (-1 / 6)
    mults = [0.25, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
    scores = [log_kde(val, train, m * scott).mean().item() for m in mults]
    best = mults[int(np.argmax(scores))]
    f = best * n ** (-1 / 6)  # the chosen multiple of Scott's factor, for all n samples
    b = torch.linspace(*beta_lim, n_grid + 1, device=z.device)
    g = torch.linspace(*gamma_lim, n_grid + 1, device=z.device)
    B, G = torch.meshgrid((b[1:] + b[:-1]) / 2, (g[1:] + g[:-1]) / 2, indexing="ij")
    th = torch.stack([B.flatten(), G.flatten()], -1)
    zg = (th.log() - sir.LOC.to(z.device)) / sir.SCALE
    wg = torch.linalg.solve_triangular(L, zg.T, upper=False).T
    logd = log_kde(wg, w, f) - (sir.SCALE * th).log().sum(-1)
    dens = (logd - logd.logsumexp(0)).exp().reshape(n_grid, n_grid).cpu().numpy()
    info = (f"{best} x Scott; held-out log-lik {max(scores):.3f} vs Scott {scores[mults.index(1.0)]:.3f}; "
            f"scores {[round(v, 3) for v in scores]}")
    return dens, f, info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("flow")
    p.add_argument("policy")
    p.add_argument("name")
    p.add_argument("--seed", type=int, default=0, help="random designs only")
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--out-dir", default="/vol/main/checkpoints/sir_problem/figures")
    p.add_argument("--title", default=None, help="name shown in panel (b) (default: NAME)")
    p.add_argument("--panel-c", choices=["samples", "grid"], default="grid",
                   help="samples: 2D histogram of posterior draws; grid: exact flow density log q(theta | h) on an "
                        "N_GRID^2 grid (backward ODE, dopri5 1e-5, exact divergence), as iDAD's grid posterior")
    p.add_argument("--grid-chunk", type=int, default=10_000, help="grid points per ODE solve")
    p.add_argument("--panel-d", choices=["none", "kde"], default="kde",
                   help="kde: add panel (d), the same posterior samples smoothed by a Gaussian KDE (kde_density)")
    args = parse_config_args(p)
    t0 = time.time()
    dev = torch.device("cuda")
    prior, guidance, _ = load_flow(args.flow, dev)
    enc = guidance.encoder
    if args.policy not in ("equal", "random"):
        pk = torch.load(args.policy, map_location=dev)
        pol = Policy(enc, SPEC.eps_dim, pk["config"]["policy_hidden_dims"], "silu", None).to(dev)
        pol.net.load_state_dict(pk["policy"])

    # iDAD's three paths as a bank of 3
    paths = json.loads(PATHS.read_text())
    ys = torch.tensor([paths[r] for r in R_LABELS], device=dev)
    bank = sir.PathBank.__new__(sir.PathBank)
    bank.ys, bank.dt_grid = ys, sir.T_END / (sir.GRID - 1)
    idx = torch.arange(3, device=dev)

    sampler = PosteriorSampler(prior, guidance, solver="dopri5", atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        if args.policy == "equal":
            tau = sir.equal_interval_taus(args.T).to(dev).expand(3, -1)
        elif args.policy == "random":
            gen = torch.Generator(device=dev).manual_seed(args.seed)
            tau = sir.random_taus(3, args.T, gen, dev).sort(dim=1).values  # order is irrelevant to the posterior
        else:
            tau, _, _ = rollout_designs(pol, enc, bank, idx, args.T)
        y = bank.observe(idx, tau)
        h_T = enc(y, sir.tau_to_eps(tau)[..., None])
        # posterior of the R = 3.0 path: N_SAMPLES draws by the forward flow ODE (stable, unlike log q on a grid far
        # from the posterior), shown as a 2D histogram in (beta, gamma) over the fixed window
        # in chunks of 500 with the solver guard (a chunk on which dopri5 hangs is redrawn)
        zs, _ = guarded_sample(sampler, N_SAMPLES, h_T[1:2])
        th_s = sir.z_to_theta(zs).cpu().numpy()
        if args.panel_d == "kde":
            tk = time.time()
            kde, kde_f, kde_info = kde_density(zs, BETA_LIM, GAMMA_LIM, N_GRID)
            print(f"kde: bandwidth factor {kde_f:.3f} ({kde_info})  ({time.time() - tk:.1f}s)", flush=True)
        if args.panel_c == "grid":
            # exact flow density on a grid of cell centres; non-finite values (the backward ODE can blow up far
            # from the posterior) -> density 0; normalised over the grid
            torch.cuda.synchronize()
            t_grid = time.time()
            b_edges, g_edges = np.linspace(*BETA_LIM, N_GRID + 1), np.linspace(*GAMMA_LIM, N_GRID + 1)
            bc = torch.tensor((b_edges[1:] + b_edges[:-1]) / 2, device=dev, dtype=torch.float32)
            gc = torch.tensor((g_edges[1:] + g_edges[:-1]) / 2, device=dev, dtype=torch.float32)
            Bc, Gc = torch.meshgrid(bc, gc, indexing="ij")
            th = torch.stack([Bc.flatten(), Gc.flatten()], -1)
            zg = (th.log() - sir.LOC.to(dev)) / sir.SCALE
            exact = PosteriorSampler(prior, guidance, solver="dopri5", atol=1e-5, rtol=1e-5, divergence="exact")
            # only the cells that hold at least one posterior sample are solved (far-out cells, q ~ 0, make dopri5
            # crawl or stall); all other cells get density 0. Guarded: points on which dopri5 stalls get density 0
            hits, _, _ = np.histogram2d(th_s[:, 0], th_s[:, 1], bins=[b_edges, g_edges])
            solve_ids = torch.as_tensor(np.flatnonzero(hits.reshape(-1) > 0), device=dev)
            logq = torch.full((zg.shape[0],), -math.inf, device=dev)
            logq[solve_ids], n_stalled = guarded_log_prob(exact, zg[solve_ids], h_T[1:2], chunk=args.grid_chunk)
            print(f"grid: solving {len(solve_ids)} of {zg.shape[0]} cells (those with >= 1 sample)", flush=True)
            n_bad = (~torch.isfinite(logq)).sum().item() - n_stalled
            logdens = torch.nan_to_num(logq, nan=-math.inf, posinf=-math.inf) - (sir.SCALE * th).log().sum(-1)
            counts = (logdens - logdens.logsumexp(0)).exp().reshape(N_GRID, N_GRID).cpu().numpy()
            torch.cuda.synchronize()
            grid_s = time.time() - t_grid
            print(f"grid: {N_GRID}x{N_GRID} points in {grid_s:.0f}s, non-finite {n_bad}, stalled {n_stalled} "
                  f"(both set to density 0)", flush=True)
    if args.panel_c == "samples":
        # 2D histogram of the samples (no smoothing); samples outside the window are counted in no bin
        b_edges, g_edges = np.linspace(*BETA_LIM, N_BINS + 1), np.linspace(*GAMMA_LIM, N_BINS + 1)
        counts, _, _ = np.histogram2d(th_s[:, 0], th_s[:, 1], bins=[b_edges, g_edges])
        in_window = counts.sum() / len(th_s)

    plt.rcParams.update({"axes.grid": True, "grid.color": "0.92", "axes.edgecolor": "0.2", "font.size": 10})
    if args.panel_d == "kde":  # 2 x 2: (a) (b) on top, (c) (d) below
        fig = plt.figure(figsize=(11, 9))
        gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.3)
        cells = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]
    else:
        fig = plt.figure(figsize=(15, 4.3))
        gs = fig.add_gridspec(1, 3, wspace=0.28)
        cells = [gs[0], gs[1], gs[2]]
    ax_a, ax_c = fig.add_subplot(cells[0]), fig.add_subplot(cells[2])

    # (a) iDAD's paths, measurements marked
    ts = torch.linspace(0, sir.T_END, sir.GRID)
    for i in range(3):
        ax_a.plot(ts, ys[i].cpu(), color=COLORS[i], lw=1.6, label=f"$R$ = {R_LABELS[i]}")
    for i in range(3):
        ax_a.scatter(tau[i].cpu(), y[i, :, 0].cpu(), zorder=3, **MARKERS[i])
    ax_a.set(xlabel="Measurement time", ylabel="Number infected", title="a) Epidemic trajectories", xlim=(-5, 105))
    ax_a.legend(loc="upper right")

    # (b) one row per experiment, t = 1 at the bottom (iDAD's layout)
    sub = cells[1].subgridspec(args.T, 1, hspace=0.08)
    rows = [fig.add_subplot(sub[k]) for k in range(args.T)]  # rows[0] is the top row: t = T
    for k, ax in enumerate(rows):
        t = args.T - k
        for i in range(3):
            ax.scatter([tau[i, t - 1].item()], [OFFSETS[i]], **MARKERS[i],
                       label=f"$R$ = {R_LABELS[i]}" if t == 1 else None)
        ax.set(xlim=(0, 100), ylim=(-0.5, 0.5), yticks=[0])
        ax.set_yticklabels([str(t)], rotation=90, va="center")
        ax.grid(axis="y", visible=False)
        if k < args.T - 1:
            ax.tick_params(axis="x", labelbottom=False)
    rows[-1].set_xlabel("Design $\\xi$")
    rows[-1].legend(loc="lower right", ncol=3, fontsize=8, handletextpad=0.2, columnspacing=0.8, framealpha=0.9)
    rows[0].set_title(f"b) {args.title or args.name} strategy, $T$ = {args.T}")
    y_mid = (rows[0].get_position().y1 + rows[-1].get_position().y0) / 2
    fig.text(rows[0].get_position().x0 - 0.028 * (15 / fig.get_figwidth()), y_mid, "Experiment number, $t$",
             rotation=90, va="center")

    # (c) posterior of the R = 3.0 path
    # grid under the posterior: transparent below 1% of the peak, grid lines drawn beneath the filled contours
    ax_c.set_axisbelow(True)
    ax_c.grid(True, color="0.85", lw=0.6)
    shown = counts.T if args.panel_c == "samples" else np.ma.masked_less(counts.T, 1e-3 * counts.max())
    ax_c.pcolormesh(b_edges, g_edges, np.ma.masked_equal(shown, 0), cmap="Blues", zorder=1)
    bl = torch.tensor(BETA_LIM)
    ax_c.plot(bl, bl / 3.0, ls=":", color="0.3", lw=1.2, label="$R$ = 3.0")
    ax_c.scatter([THETAS[1][0]], [THETAS[1][1]], marker="x", color="red", s=80, label="True value", zorder=3)
    ax_c.set(xlabel="Infection rate, $\\beta$", ylabel="Recovery rate, $\\gamma$", xlim=BETA_LIM, ylim=GAMMA_LIM,
             title=f"c) Posterior samples, $R$ = 3.0 ({100 * in_window:.0f}% in window)" if args.panel_c == "samples"
             else "c) Posterior density (grid), $R$ = 3.0")
    ax_c.legend(loc="lower right")

    if args.panel_d == "kde":
        # (d) the same samples, smoothed: KDE density on the grid, shown as panel (c) of the grid version
        ax_d = fig.add_subplot(cells[3])
        ax_d.set_axisbelow(True)
        ax_d.grid(True, color="0.85", lw=0.6)
        kb, kg = np.linspace(*BETA_LIM, N_GRID + 1), np.linspace(*GAMMA_LIM, N_GRID + 1)
        ax_d.pcolormesh(kb, kg, np.ma.masked_less(kde.T, 1e-3 * kde.max()), cmap="Blues", zorder=1)
        ax_d.plot(bl, bl / 3.0, ls=":", color="0.3", lw=1.2, label="$R$ = 3.0")
        ax_d.scatter([THETAS[1][0]], [THETAS[1][1]], marker="x", color="red", s=80, label="True value", zorder=3)
        ax_d.set(xlabel="Infection rate, $\\beta$", ylabel="Recovery rate, $\\gamma$", xlim=BETA_LIM, ylim=GAMMA_LIM,
                 title="d) Posterior, KDE of the samples")
        ax_d.legend(loc="lower right")

    suffix = ("_grid" if args.panel_c == "grid" else "") + ("_kde" if args.panel_d == "kde" else "")
    out = Path(args.out_dir) / f"fig5_sir_{args.name}{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out, f"({time.time() - t0:.0f}s) designs", [[round(v, 1) for v in r] for r in tau.cpu().tolist()])


if __name__ == "__main__":
    main()
