"""What the dy/dtau window does to SIR paths: smoothed paths and slopes vs the true slope, good / mild / bad cases.

The central difference over +-h, (I(t + h) - I(t - h)) / 2h, is exactly the slope of the path after a moving average
over [t - h, t + h]; the figure shows those smoothed paths and their slopes for h = 0.01 (one grid step, the training
default), 0.1 and 0.5, against the true slope = the drift beta S I / N - gamma I at the path's state (what the path does
apart from its random kicks). Paths: --n fresh prior draws (seed 777, the SDE of sir.py, S kept), filtered as sir.py
(mean I >= 1). Error of a window on a path: RMS(slope_h - drift) / RMS(drift) over the active epidemic (I >= 1).
Cases by the error of the +-0.5 window: good = 2 paths near its 10% quantile, mild = 2 near the median, bad = the 2
largest of the paths at its 99% quantile (most-smoothed peaks). Columns: the path and its smoothed versions, the
slopes over the epidemic, the slopes zoomed on the steepest part.
    window_paths.py [--n 2000] [--out FIG]
"""

from __future__ import annotations

import argparse
import math

import matplotlib
import numpy as np
import torch

from flow_bed.config import parse_args as parse_config_args
from flow_bed.simulators import sir

matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = [0.01, 0.1, 0.5]
COL = {0.01: "#b8bec6", 0.1: "#2a6fdb", 0.5: "#d08c00"}


@torch.no_grad()
def simulate(n, gen, dev):
    dt_grid = sir.T_END / (sir.GRID - 1)
    dt = dt_grid / sir.SUBSTEPS
    z = torch.randn(n, 2, device=dev, generator=gen)
    beta, gamma = sir.z_to_theta(z).unbind(-1)
    S = torch.full((n,), sir.N_POP - sir.I0, device=dev)
    I = torch.full((n,), sir.I0, device=dev)
    Ss, Is = torch.empty(n, sir.GRID, device=dev), torch.empty(n, sir.GRID, device=dev)
    Ss[:, 0], Is[:, 0] = S, I
    sq = math.sqrt(dt)
    for k in range(1, sir.GRID):
        for _ in range(sir.SUBSTEPS):
            S.clamp_(0.0, sir.N_POP)
            I.clamp_(0.0, sir.N_POP)
            p_inf = beta * S * I / sir.N_POP
            p_rec = gamma * I
            dw = torch.randn(2, n, device=dev, generator=gen) * sq
            a, r = p_inf.sqrt() * dw[0], p_rec.sqrt() * dw[1]
            S, I = S - p_inf * dt - a, I + (p_inf - p_rec) * dt + a - r
        Ss[:, k], Is[:, k] = S.clamp(0, sir.N_POP), I.clamp(0, sir.N_POP)
    return beta, gamma, Ss, Is


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="/vol/main/checkpoints/sir_problem/figures/window_paths.png")
    args = parse_config_args(p)
    dev = torch.device("cuda")
    beta, gamma, S, I = simulate(args.n, torch.Generator(device=dev).manual_seed(777), dev)
    keep = (I.mean(1) >= 1).nonzero()[:, 0]
    beta, gamma, S, I = beta[keep], gamma[keep], S[keep], I[keep]
    dtg = sir.T_END / (sir.GRID - 1)
    t = torch.arange(sir.GRID, device=dev) * dtg
    drift = beta[:, None] * S * I / sir.N_POP - gamma[:, None] * I
    kk = torch.arange(sir.GRID, device=dev)
    slope, smooth = {}, {}
    cs = torch.cat([torch.zeros(I.shape[0], 1, device=dev), I.cumsum(1)], 1)  # prefix sums for the moving average
    for h in H:
        w = max(1, round(h / dtg))
        kp, km = (kk + w).clamp(max=sir.GRID - 1), (kk - w).clamp(min=0)
        slope[h] = (I[:, kp] - I[:, km]) / ((kp - km) * dtg)
        smooth[h] = (cs[:, kp + 1] - cs[:, km]) / (kp - km + 1)
    active = I >= 1
    err = {}
    for h in H:
        num = (((slope[h] - drift) ** 2) * active).sum(1) / active.sum(1)
        den = ((drift ** 2) * active).sum(1) / active.sum(1)
        err[h] = (num / den).sqrt().cpu().numpy()
    e5 = err[0.5]
    print(f"{I.shape[0]} paths (after the filter). Relative slope error RMS(slope - drift) / RMS(drift), quantiles "
          f"10/50/90/99%:")
    for h in H:
        q = np.quantile(err[h], [0.1, 0.5, 0.9, 0.99])
        print(f"  +-{h:<5g} {' '.join(f'{v:7.3f}' for v in q)}")
    print(f"  +-0.1 better than +-0.5 on {100 * np.mean(err[0.1] < err[0.5]):.0f}% of paths")
    r0 = (beta / gamma).cpu().numpy()
    for lab, lo, hi in (("R0 < 3", 0, 3), ("3-8", 3, 8), ("> 8", 8, 1e9)):
        m = (r0 >= lo) & (r0 < hi)
        print(f"  R0 {lab:6s} ({m.sum():4d} paths): median error +-0.01 {np.median(err[0.01][m]):.3f}, +-0.1 "
              f"{np.median(err[0.1][m]):.3f}, +-0.5 {np.median(err[0.5][m]):.3f}")

    order = np.argsort(e5)
    n = len(order)
    pick = {"good": [order[int(0.10 * n)], order[int(0.10 * n) + 1]],
            "mild": [order[int(0.50 * n)], order[int(0.50 * n) + 1]],
            "bad": [order[int(0.99 * n)], order[-1]]}
    rows = [(lab, i) for lab, ids in pick.items() for i in ids]
    fig, axes = plt.subplots(len(rows), 3, figsize=(20, 3.3 * len(rows)))
    tc = t.cpu().numpy()
    for r, (lab, i) in enumerate(rows):
        Ii, di = I[i].cpu().numpy(), drift[i].cpu().numpy()
        act = np.nonzero(Ii >= 1)[0]
        t_end = min(100.0, tc[act[-1]] + 3) if act.size else 100.0
        sel = tc <= t_end
        b, g = beta[i].item(), gamma[i].item()
        # 1: path and smoothed paths
        ax = axes[r, 0]
        ax.plot(tc[sel], Ii[sel], color="#3d4a5c", lw=0.7, label="path I(t)")
        for h in (0.1, 0.5):
            ax.plot(tc[sel], smooth[h][i].cpu().numpy()[sel], color=COL[h], lw=1.3, label=f"moving average +-{h}")
        ax.set_ylabel(f"{lab}\nI(t)")
        ax.set_title(f"beta {b:.2f}, gamma {g:.3f}, R0 {b / g:.1f}; slope error +-0.01 {err[0.01][i]:.2f}, "
                     f"+-0.1 {err[0.1][i]:.2f}, +-0.5 {err[0.5][i]:.2f}", fontsize=9, loc="left")
        # 2: slopes over the epidemic
        ax = axes[r, 1]
        for h in H:
            ax.plot(tc[sel], slope[h][i].cpu().numpy()[sel], color=COL[h], lw=0.6 if h == 0.01 else 1.2,
                    label=f"slope +-{h}" + (" (training)" if h == 0.01 else ""))
        ax.plot(tc[sel], di[sel], color="black", lw=1.4, label="true slope (drift)")
        lo, hi = di[sel].min(), di[sel].max()
        pad = 0.6 * (hi - lo + 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel("dI/dt")
        # 3: zoom on the steepest part
        ax = axes[r, 2]
        k0 = int(np.argmax(np.abs(di)))
        zsel = (tc >= tc[k0] - 3) & (tc <= tc[k0] + 3)
        for h in H:
            ax.plot(tc[zsel], slope[h][i].cpu().numpy()[zsel], color=COL[h], lw=0.6 if h == 0.01 else 1.5)
        ax.plot(tc[zsel], di[zsel], color="black", lw=1.6)
        lo, hi = di[zsel].min(), di[zsel].max()
        pad = 0.6 * (hi - lo + 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title("zoom: +-3 time units around the steepest true slope", fontsize=9, loc="left")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    for c in range(3):
        axes[-1, c].set_xlabel("time")
    fig.suptitle("dy/dtau windows on SIR paths: the +-h central difference = the slope of the path's moving average over "
                 "[t-h, t+h]; rows: 2 good / 2 mild / 2 bad cases by the +-0.5 slope error (grey +-0.01 runs off the "
                 "axis)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=105)
    print(f"saved {args.out}")

    # second figure: the same cases, one column per window (own y-scale), plus the path zoomed with its moving averages
    fig, axes = plt.subplots(len(rows), 4, figsize=(24, 3.3 * len(rows)))
    for r, (lab, i) in enumerate(rows):
        Ii, di = I[i].cpu().numpy(), drift[i].cpu().numpy()
        act = np.nonzero(Ii >= 1)[0]
        t_end = min(100.0, tc[act[-1]] + 3) if act.size else 100.0
        sel = tc <= t_end
        b, g = beta[i].item(), gamma[i].item()
        k0 = int(np.argmax(np.abs(di)))
        zsel = (tc >= tc[k0] - 1.5) & (tc <= tc[k0] + 1.5)
        ax = axes[r, 0]
        ax.plot(tc[zsel], Ii[zsel], color="#3d4a5c", lw=0.8, label="path I(t)")
        for h in H:
            ax.plot(tc[zsel], smooth[h][i].cpu().numpy()[zsel], color=COL[h] if h != 0.01 else "#7d8793",
                    lw=1.3, label=f"moving average +-{h}")
        ax.set_ylabel(f"{lab}\nI(t)")
        ax.set_title(f"beta {b:.2f}, gamma {g:.3f}, R0 {b / g:.1f}: path, zoom +-1.5 around the steepest slope",
                     fontsize=9, loc="left")
        for c, h in enumerate(H, start=1):
            ax = axes[r, c]
            s = slope[h][i].cpu().numpy()[sel]
            ax.plot(tc[sel], s, color=COL[h] if h != 0.01 else "#7d8793", lw=0.6 if h == 0.01 else 1.0,
                    label=f"slope +-{h}")
            ax.plot(tc[sel], di[sel], color="black", lw=1.4, label="true slope (drift)")
            lo, hi = np.quantile(s, [0.005, 0.995])
            lo, hi = min(lo, di[sel].min()), max(hi, di[sel].max())
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.set_title(f"window +-{h}{' (training)' if h == 0.01 else ''}: slope error {err[h][i]:.2f}",
                         fontsize=9, loc="left")
            if c == 1:
                ax.set_ylabel("dI/dt")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    for c in range(1, 4):
        axes[0, c].legend(fontsize=8, loc="upper right")
    for c in range(4):
        axes[-1, c].set_xlabel("time")
    fig.suptitle("The same 6 paths, one column per dy/dtau window (each on its own scale): slope estimate vs the true "
                 "slope; first column: the path near its steepest point with the three moving averages", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out2 = args.out.replace(".png", "_by_window.png")
    fig.savefig(out2, dpi=100)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
