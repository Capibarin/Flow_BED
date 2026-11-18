"""Basins of the finalq objective (q = 5, final term only), without a policy network: optimise one static sequence of 5
design times directly, from many starts, with exactly finalq's training gradient, and see where the starts end up.

Parameters: u (S starts, 5), designs by iDAD's increasing transform tau_t = tau_{t-1} + sigmoid(u_t) (UB - tau_{t-1}).
Loss (as train_sir_policy.py, --policy-term final_random, q_min = Q = 5): the divergence of the guidance at the full
5-step history, batch --batch paths of a training path bank (seed 1, --bank, re-simulated every --refresh steps), one
flow time per step (shared by the batch), d y / d tau = central difference over +- --dy-window (straight-through).
Adam on u, lr cosine from --lr to --lr / 50 over --steps.
Starts: 'untrained' (sigmoid(u) = 0.5: ~50, 75, 87, 94, 97), 'early' (0.15: ~15, 28, 39, 48, 56), 'idad' (iDAD seed
42's mean designs), and --random starts with fractions sigmoid(u_t) ~ U(0.03, 0.9).
--t-mode: flow times of the divergence (batch = training; sample; strat8 = 8 stratified per sample, averaged);
--loss logq: the exact -log q(z | h_5) of the flow (rk4, --ode-steps, backprop through the solver) instead.
Checkpoints (every --every steps): BA = E[log q(z | h_5)] - E[log p(z)] on --eval held-out paths (seed 12345;
rk4, 20 steps; the same paths for all starts) and the gradient's signal-to-noise ratio: per-path gradients of the loss
w.r.t. u on those paths (one random flow time per path), |mean| / s.e. per component, averaged over the 5 components,
reported for a batch of 256 (= SNR on --eval paths x sqrt(256 / eval)).
Basins: final designs clustered (a start joins a cluster when every design is within --tol time units of the
cluster's best start); per cluster: starts, designs, BA.
    sir_static_basins.py [--dy-window 0.01] [--tag h001] ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch

from flow_bed.config import parse_args as parse_config_args
from flow_bed.models import PosteriorSampler
from flow_bed.models.posterior_sampler import exact_divergence
from flow_bed.simulators import sir
from flow_bed.sir import load_flow

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = 5


def designs_of(u):
    """u (..., T) -> increasing designs (..., T)."""
    cols, prev = [], torch.full(u.shape[:-1], sir.LB, device=u.device)
    for t in range(T):
        prev = sir.next_tau(u[..., t], prev)
        cols.append(prev)
    return torch.stack(cols, -1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow", default="sir_flow_S_silu.pt")
    p.add_argument("--dy-window", type=float, default=0.01)
    p.add_argument("--random", type=int, default=32)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--bank", type=int, default=25_000)
    p.add_argument("--refresh", type=int, default=2500)
    p.add_argument("--every", type=int, default=250)
    p.add_argument("--eval", type=int, default=1024)
    p.add_argument("--rows", type=int, default=8192, help="rows per forward chunk")
    p.add_argument("--tol", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="h001")
    p.add_argument("--t-mode", choices=["batch", "sample", "strat8"], default="batch",
                   help="flow times of the divergence loss: one per batch (training), one per sample, 8 stratified "
                        "per sample (averaged)")
    p.add_argument("--loss", choices=["div", "logq"], default="div",
                   help="div: the training divergence; logq: -log q(z | h_5) by the flow ODE (rk4, --ode-steps)")
    p.add_argument("--ode-steps", type=int, default=10)
    p.add_argument("--mem-frac", type=float, default=None, help="cap this process's GPU memory (fraction of total)")
    p.add_argument("--idad", default="/vol/main/checkpoints/sir_problem/idad/idad_designs_s42.pt")
    p.add_argument("--fig-dir", default="/vol/main/checkpoints/sir_problem/figures")
    args = parse_config_args(p)
    dev = torch.device("cuda")
    if args.mem_frac is not None:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, torch.cuda.current_device())
    torch.manual_seed(args.seed)

    prior, guidance, _ = load_flow(args.flow, dev)
    enc = guidance.encoder
    post = PosteriorSampler(prior, guidance, solver="rk4", n_steps=20, divergence="exact")
    post_train = PosteriorSampler(prior, guidance, solver="rk4", n_steps=args.ode_steps, divergence="exact")
    prs = PosteriorSampler(prior, None, solver="rk4", n_steps=20, divergence="exact")
    train_bank = sir.PathBank(args.bank, 1, dev)
    held = sir.PathBank(50_000, 12345, dev)
    g0 = torch.Generator(device=dev).manual_seed(0)
    ev = held.valid[torch.randperm(held.valid.shape[0], device=dev, generator=g0)][: args.eval]
    with torch.no_grad():
        logp_ev = prs.log_prob(held.z[ev])
    w = max(1, round(args.dy_window / train_bank.dt_grid))

    # starts
    def u_of_taus(taus):
        taus, prev, us = torch.as_tensor(taus, dtype=torch.float32), sir.LB, []
        for tv in taus.tolist():
            f = (tv - prev) / (sir.UB - prev)
            us.append(math.log(f / (1 - f)))
            prev = tv
        return us
    idad_mean = torch.load(args.idad, map_location="cpu")["tau"][: args.eval].float().mean(0)
    names = ["untrained", "early", "idad"] + [f"r{i}" for i in range(args.random)]
    gr = torch.Generator().manual_seed(args.seed)
    fr = 0.03 + torch.rand(args.random, T, generator=gr) * 0.87
    u0 = torch.cat([torch.zeros(1, T), torch.full((1, T), math.log(0.15 / 0.85)), torch.tensor([u_of_taus(idad_mean)]),
                    torch.log(fr / (1 - fr))]).to(dev)
    S = u0.shape[0]
    u = u0.clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr / 50)

    def loss_chunks(bank, idx_b, z_b, e_b, t_b, uu, per_path=False):
        """sum over starts of the mean divergence; uu (S, T) or (S, B, T) with per_path."""
        S_, B = uu.shape[0], idx_b.shape[0]
        per = max(1, args.rows // B)
        total = 0.0
        for s0 in range(0, S_, per):
            us = uu[s0: s0 + per]
            n = us.shape[0]
            tau = designs_of(us if per_path else us[:, None, :].expand(n, B, T)).reshape(n * B, T)
            idx = idx_b.repeat(n)
            y = bank.observe(idx, tau, w)
            h = enc(y, sir.tau_to_eps(tau)[..., None])
            tt = t_b.repeat(n)
            x = (tt[:, None] * z_b.repeat(n, 1) + prior.lambda_t(tt)[:, None] * e_b.repeat(n, 1)).requires_grad_(True)
            if args.loss == "logq":
                d = -post_train.log_prob(z_b.repeat(n, 1), h)
            else:
                d = exact_divergence(guidance(x, tt, h), x, create_graph=True)
            part = d.sum() if per_path else d.view(n, B).mean(1).sum()
            part.backward()
            total += part.item()
        return total

    @torch.no_grad()
    def ba(uu):
        out = []
        tau_all = designs_of(uu)
        for s in range(uu.shape[0]):
            tau = tau_all[s].expand(args.eval, T)
            h = enc(held.observe(ev, tau), sir.tau_to_eps(tau)[..., None])
            out.append(torch.nan_to_num(post.log_prob(held.z[ev], h), nan=-1e4, neginf=-1e4) - logp_ev)
        return torch.stack(out)  # (S, eval)

    def snr(uu):
        up = uu.detach()[:, None, :].expand(S, args.eval, T).clone().requires_grad_(True)
        gen = torch.Generator(device=dev).manual_seed(5)
        e = torch.randn(args.eval, 2, device=dev, generator=gen)
        tt = torch.rand(args.eval, device=dev, generator=gen)
        loss_chunks(held, ev, held.z[ev], e, tt, up, per_path=True)
        g = up.grad  # (S, eval, T)
        r = g.mean(1).abs() / (g.std(1) / math.sqrt(args.eval))
        return (r * math.sqrt(256 / args.eval)).mean(1).cpu(), g.abs().mean((1, 2)).cpu()

    hist = {"step": [], "ba": [], "tau": [], "snr": [], "gabs": []}

    def checkpoint(step):
        b = ba(u.detach())
        sn, ga = snr(u)
        hist["step"].append(step)
        hist["ba"].append(b.mean(1).cpu())
        hist["tau"].append(designs_of(u.detach()).cpu())
        hist["snr"].append(sn)
        hist["gabs"].append(ga)
        print(f"step {step:5d}: BA mean {b.mean().item():.3f}, best {b.mean(1).max().item():.3f}, worst "
              f"{b.mean(1).min().item():.3f}; SNR(256) median {sn.median().item():.2f}; untrained "
              f"{[round(v, 1) for v in hist['tau'][-1][0].tolist()]} BA {hist['ba'][-1][0]:.3f}", flush=True)
        return b

    for step in range(args.steps + 1):
        if step % args.every == 0:
            b_last = checkpoint(step)
        if step == args.steps:
            break
        if step > 0 and step % args.refresh == 0:
            train_bank.refresh()
        idx = train_bank.sample_idx(args.batch)
        z = train_bank.z[idx]
        e = torch.randn_like(z)
        if args.t_mode == "batch":
            tt = torch.rand(1, device=dev).expand(args.batch)
        elif args.t_mode == "sample":
            tt = torch.rand(args.batch, device=dev)
        else:  # 8 stratified flow times per sample: rows repeated 8 times
            K = 8
            idx, z = idx.repeat(K), z.repeat(K, 1)
            e = torch.randn_like(z)
            tt = (torch.arange(K, device=dev).repeat_interleave(args.batch) + torch.rand(K * args.batch, device=dev)) / K
        opt.zero_grad()
        loss_chunks(train_bank, idx, z, e, tt, u)
        opt.step()
        sched.step()

    # basins
    tau_f = hist["tau"][-1].numpy()
    ba_f = hist["ba"][-1].numpy()
    se_f = (b_last.std(1) / math.sqrt(args.eval)).cpu().numpy()
    order = np.argsort(-ba_f)
    clusters = []
    for s in order:
        for c in clusters:
            if np.abs(tau_f[s] - tau_f[c["members"][0]]).max() < args.tol:
                c["members"].append(int(s))
                break
        else:
            clusters.append({"members": [int(s)]})
    print(f"\n=== basins (window +-{args.dy_window}; {S} starts; tolerance {args.tol} time units)")
    print(f"{'#':>2s} {'starts':>6s} {'BA (best start)':>16s} {'BA range':>16s}  designs (best start)   named starts")
    for i, c in enumerate(clusters):
        m = c["members"]
        c.update({"n": len(m), "designs": tau_f[m[0]].tolist(), "ba": float(ba_f[m[0]]), "ba_se": float(se_f[m[0]]),
                  "ba_min": float(ba_f[m].min()), "ba_max": float(ba_f[m].max()),
                  "named": [names[k] for k in m if not names[k].startswith("r")],
                  "start_designs": [hist["tau"][0][k].tolist() for k in m]})
        print(f"{i:2d} {len(m):6d} {c['ba']:9.3f}+-{c['ba_se']:.3f} {c['ba_min']:7.3f}..{c['ba_max']:6.3f}  "
              f"{[round(v, 1) for v in c['designs']]}   {c['named']}")
    for nm in ("untrained", "early", "idad"):
        k = names.index(nm)
        print(f"start {nm:9s}: {[round(v, 1) for v in hist['tau'][0][k].tolist()]} BA {hist['ba'][0][k]:.3f} -> "
              f"{[round(v, 1) for v in hist['tau'][-1][k].tolist()]} BA {hist['ba'][-1][k]:.3f}")
    print("SNR (batch 256) by checkpoint, median over starts: "
          + ", ".join(f"{st}: {sn.median():.2f}" for st, sn in zip(hist["step"], hist["snr"])))
    # basin of each start: SNR at the start and at the end, per cluster
    for i, c in enumerate(clusters):
        m = c["members"]
        print(f"basin {i}: SNR start median {hist['snr'][0][m].median():.2f}, end median {hist['snr'][-1][m].median():.2f}; "
              f"|grad| start {hist['gabs'][0][m].median():.3g}, end {hist['gabs'][-1][m].median():.3g}")

    # figure: BA trajectories coloured by basin; start -> end designs
    cols = ["#2a6fdb", "#d1495b", "#d08c00", "#2e9e6a", "#7b3fa0", "#5c6670", "#e377c2", "#17becf"]
    cl_of = {k: i for i, c in enumerate(clusters) for k in c["members"]}
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    steps = hist["step"]
    for k in range(S):
        col = cols[min(cl_of[k], len(cols) - 1)]
        ax[0].plot(steps, [b[k] for b in hist["ba"]], color=col, lw=2.5 if not names[k].startswith("r") else 0.8,
                   alpha=0.9)
        ax[1].plot(range(1, T + 1), hist["tau"][-1][k], "-o", color=col, lw=0.8, ms=3)
        ax[2].plot(steps, [sn[k] for sn in hist["snr"]], color=col, lw=0.8)
    for nm in ("untrained", "early", "idad"):
        k = names.index(nm)
        ax[0].annotate(nm, (steps[-1], hist["ba"][-1][k]), fontsize=8, xytext=(3, 0), textcoords="offset points")
    ax[0].set(xlabel="optimisation step", ylabel="BA (held-out, nats)", title="BA of each start (colour = final basin)")
    ax[1].set(xlabel="design index", ylabel="design time", title="final designs (colour = basin)")
    ax[2].set(xlabel="optimisation step", ylabel="SNR of the batch gradient (batch 256)", title="gradient signal / noise")
    ax[2].axhline(1, color="#9aa1a9", ls="--", lw=1)
    handles = [plt.Line2D([], [], color=cols[min(i, len(cols) - 1)], lw=2,
                          label=f"basin {i}: {c['n']} starts, BA {c['ba']:.2f}") for i, c in enumerate(clusters[:8])]
    ax[1].legend(handles=handles, fontsize=8)
    fig.suptitle(f"finalq (q = 5) objective, static designs optimised directly; dy/dtau window +-{args.dy_window}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{args.fig_dir}/static_basins_{args.tag}.png", dpi=110)
    out = {"args": vars(args), "names": names, "clusters": clusters,
           "hist": {k: [x.tolist() if torch.is_tensor(x) else x for x in v] for k, v in hist.items()}}
    Path(f"static_basins_{args.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
