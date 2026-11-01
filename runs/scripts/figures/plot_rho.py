"""Figure of the per-tau design-gradient RMS (analysis/tau_rms.py) and the variance-optimal tau density rho ∝ RMS.

rho is piecewise linear on the measured grid; sample_tau draws from it by inverse CDF (quadratic within a segment).
Variance gain of rho over U[0, 1] for the second moment of the importance-weighted gradient:
    E_U ||g||^2 = ∫ m2,   E_rho ||g / rho||^2 = ∫ m2 / rho   (= (∫ sqrt m2)^2 at the optimum)
Prints the quantiles of rho (designs of the policy, q ~ U{1..30}) and the gains.

    python runs/scripts/figures/plot_rho.py --rms checkpoints/tau_rms.json --out checkpoints/figures_v2/tau_rho.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flow_bed.lf.tau_density import rho_nodes, sample_tau

COL = {"1": "#2a78d6", "10": "#eb6834", "30": "#1baf7a", "mix": "#0b0b0b"}
LAB = {"1": "n = 1", "10": "n = 10", "30": "n = 30", "mix": "q ~ U{1..30} (training)"}
INK2, GRID = "#52514e", "#e4e3df"


def gain(taus, m2, rho):
    """E_U ||g||^2 / E_rho ||g / rho||^2, on a fine grid with m2 and rho interpolated linearly."""
    t = np.linspace(0, 1, 200_001)
    m, r = np.interp(t, taus, m2), np.interp(t, taus, rho)
    return np.trapezoid(m, t) / np.trapezoid(np.where(r > 0, m / np.where(r > 0, r, 1), 0), t)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rms", type=Path, required=True, help="JSON written by analysis/tau_rms.py")
    p.add_argument("--out", type=Path, required=True, help="figure (png)")
    args = p.parse_args()
    with open(args.rms) as f:
        D = json.load(f)
    taus, R, meta = np.array(D["taus"]), D["results"], D.get("meta", {})

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": "#0b0b0b", "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")

    # (a) RMS per tau
    a = ax[0, 0]
    for nk, col in COL.items():
        for kind, ls in [("policy", "-"), ("random", "--")]:
            r = R[f"{kind}_{nk}"]["hu"]
            a.plot(taus[:-1], np.array(r["rms"])[:-1], ls, color=col, lw=2 if nk == "mix" else 1.5, marker="o",
                   ms=3, label=f"{LAB[nk]}" if kind == "policy" else None)
    a.plot([], [], "-", color=INK2, label="policy designs")
    a.plot([], [], "--", color=INK2, label="random designs")
    a.set_yscale("log")
    a.set_xlabel(r"$\tau$")
    a.set_ylabel(r"RMS  $\sqrt{E\|\partial_\xi \,\mathrm{div}\, w(x_\tau,\tau,h)\|^2}$")
    a.set_title(r"(a) design-gradient RMS per $\tau$ (Hutchinson; 0 at $\tau = 1$)", loc="left")
    a.grid(color=GRID, lw=0.6)
    a.legend(frameon=False, fontsize=8)

    # (b) rho on [0, 1] and (c) zoom on the data end
    main_rho = rho_nodes(taus, np.array(R["policy_mix"]["hu"]["rms"]))
    for axx, xlim, title in [(ax[0, 1], (0, 1), r"(b) $\rho(\tau) \propto$ RMS, normalised on [0, 1]"),
                             (ax[1, 0], (0.85, 1), r"(c) zoom: $\tau \in [0.85, 1]$")]:
        for key, ls, lab in [("policy_mix", "-", "policy designs, q ~ U{1..30}"),
                             ("random_mix", "--", "random designs, q ~ U{1..30}")]:
            for d, lw, alpha in [("hu", 2.2, 1.0), ("ex", 1.2, 0.6)]:
                rho = rho_nodes(taus, np.array(R[key][d]["rms"]))
                axx.plot(taus, rho, ls, color=COL["mix"] if key == "policy_mix" else COL["1"], lw=lw, alpha=alpha,
                         marker="o" if d == "hu" else None, ms=3,
                         label=f"{lab} ({'Hutchinson' if d == 'hu' else 'exact div'})")
        for nk in ["1", "10", "30"]:
            axx.plot(taus, rho_nodes(taus, np.array(R[f"policy_{nk}"]["hu"]["rms"])), "-", color=COL[nk], lw=1,
                     alpha=0.8, label=f"policy designs, {LAB[nk]}")
        axx.axhline(1, color=INK2, lw=1, ls=":", label=r"uniform $\rho \equiv 1$")
        axx.set_xlim(*xlim)
        axx.set_xlabel(r"$\tau$")
        axx.set_ylabel(r"$\rho(\tau)$")
        axx.set_title(title, loc="left")
        axx.grid(color=GRID, lw=0.6)
    ax[0, 1].legend(frameon=False, fontsize=7, loc="upper left")

    # (d) inverse-CDF samples of the training-mixture rho, and the variance gains
    d = ax[1, 1]
    taus_s = sample_tau(taus, main_rho, np.random.default_rng(0).random(400_000))
    d.hist(taus_s, bins=200, range=(0, 1), density=True, color="#86b6ef", label="400k inverse-CDF samples")
    d.plot(taus, main_rho, "-", color=COL["mix"], lw=1.5, marker="o", ms=3, label=r"$\rho$ (policy designs, q ~ U{1..30})")
    d.set_xlabel(r"$\tau$")
    d.set_ylabel("density")
    d.set_title(r"(d) sampler check, and gain $\int m_2 \,/\, \int m_2/\rho$ of $\rho$ over U[0, 1]", loc="left")
    d.grid(color=GRID, lw=0.6)
    d.legend(frameon=False, fontsize=8, loc="upper left")
    lines = []
    for kind in ["policy", "random"]:
        for nk in COL:
            r = R[f"{kind}_{nk}"]["hu"]
            m2 = np.array(r["m2"])
            own = gain(taus, m2, rho_nodes(taus, np.array(r["rms"])))
            lines.append(f"{kind:6s} {LAB[nk].split(' (')[0]:13s} own rho x{own:5.1f}   mix rho "
                         f"x{gain(taus, m2, main_rho):5.1f}")
    d.text(0.02, 0.62, "\n".join(lines), transform=d.transAxes, family="monospace", fontsize=7, color="#0b0b0b",
           va="top")
    n_pt = meta.get("batch", 0) * meta.get("batches", 0)
    n_mix = meta.get("batch", 0) * meta.get("batches_mix", 0)
    fig.suptitle("Variance-optimal flow-time density for the divergence policy gradient, "
                 r"$\rho(\tau) \propto \sqrt{E\|\partial_\xi\,\mathrm{div}\,w\|^2}$" "\n"
                 f"flow {Path(meta.get('flow_ckpt', '?')).name}, policy {Path(meta.get('policy', '?')).name}; "
                 f"{len(taus)} grid points, {n_pt} samples per point ({n_mix} for q ~ U{{1..30}})", fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print("saved", args.out)
    q = np.quantile(taus_s, [0.1, 0.25, 0.5, 0.75, 0.9])
    print("rho (policy, mix) quantiles 10/25/50/75/90%:", np.round(q, 3))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
