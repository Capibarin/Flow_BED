"""SIR evaluation on a held-out path bank (seed 12345, never used in training).

Designs: equal (iDAD equal interval), random (iDAD random baseline), NAME=policy.pt (rollouts of a policy), or
NAME=file:DESIGNS.pt (designs saved per run for these runs, e.g. iDAD's policy by idad_designs.py: dict with "tau"
or a tensor (runs, T)).
For each, on the same --runs test paths (paired across designs):
  InfoNCE   lower bound on the EIG with the flow as critic U = log q(z | h) - log p(z) (valid for any critic, as iDAD's
            evaluation with its own critic), --L contrastive z from the test bank's prior draws (shared by all runs),
            log q / log p by the flow ODE (--solver, --n-steps).
  BA        E[log q(z | h)] - E[log p(z)] (log p = the prior flow; an estimate, not a strict bound).
  per run   with --out-json, the per-run InfoNCE / BA terms and designs of every design method go to <out-json>.pt.
  coverage  (--calib runs) fraction of runs whose true z lies in the posterior's 50 / 80 / 95% highest-density
            region (from --calib-samples posterior samples and their log q); ideal = the level.

Solvers: --solver rk4 (default, --n-steps) is the routine evaluation: fast, the same critic for every design set, so
the comparisons are fair (the bound is valid for any critic). With an adaptive solver (--solver dopri5), the final
scoring: the backward ODE can crawl or hang on pairs far outside the posterior, so the contrastive terms are first
screened with rk4 (--n-steps) and only the pairs within --screen-margin nats of their run's best term are re-solved
with dopri5 under the freeze guard (flow_bed.models.guarded_sampling.frozen_log_prob: a point that leaves the ball
|x| <= --freeze-radius or turns non-finite is frozen and gets density 0); the dropped pairs keep their rk4 value
(< e^-margin of the row's logsumexp). True pairs are all solved with the guarded dopri5. Coverage stays on rk4
(forward sampling with an adaptive solver can stall; it is a secondary diagnostic).
--config file.yaml sets the defaults of these arguments (see runs/configs/sir/eval_infonce.yaml and flow_bed.config).
    python runs/scripts/evaluation/eval_sir_infonce.py --config runs/configs/sir/eval_infonce.yaml \\
        --designs equal NAME=POLICY.pt --out-json OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from flow_bed.config import parse_args
from flow_bed.models import PosteriorSampler
from flow_bed.models.guarded_sampling import frozen_log_prob
from flow_bed.models.policy import Policy
from flow_bed.simulators import sir
from flow_bed.sir import SPEC, load_flow, rollout_designs

ADAPTIVE = ("dopri5", "tsit5")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt", type=Path, default=None, help="required, as a flag or in the config")
    p.add_argument("--designs", nargs="+", default=["equal", "random"])
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--runs", type=int, default=4096)
    p.add_argument("--L", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=100_000, help="pairs per ODE solve")
    p.add_argument("--solver", default="rk4")
    p.add_argument("--n-steps", type=int, default=20, help="rk4 steps (also the screening pass of a final scoring)")
    p.add_argument("--screen-margin", type=float, default=30.0,
                   help="adaptive solver only: re-solve the contrastive pairs within this many nats of their run's "
                        "best rk4 term; the rest keep the rk4 value")
    p.add_argument("--freeze-radius", type=float, default=20.0, help="adaptive solver only: freeze ball |x| <= R")
    p.add_argument("--freeze-cap", type=int, default=3000, help="adaptive solver only: vector-field call backstop")
    p.add_argument("--calib", type=int, default=1000)
    p.add_argument("--calib-samples", type=int, default=1000)
    p.add_argument("--bank", type=int, default=50_000)
    p.add_argument("--out-json", type=Path, default=None)
    args = parse_args(p)
    if args.flow_ckpt is None:
        p.error("--flow-ckpt is required (as a flag or in the config)")
    dev = torch.device("cuda")
    adaptive = args.solver in ADAPTIVE
    prior, guidance, _ = load_flow(args.flow_ckpt, dev)
    enc = guidance.encoder
    sampler = PosteriorSampler(prior, guidance, solver=args.solver, n_steps=args.n_steps, divergence="exact")
    prior_sampler = PosteriorSampler(prior, None, solver=args.solver, n_steps=args.n_steps, divergence="exact")
    # screening and coverage stay on rk4 when the final solver is adaptive
    screen = (PosteriorSampler(prior, guidance, solver="rk4", n_steps=args.n_steps, divergence="exact")
              if adaptive else sampler)

    def log_prob(s, z, h):
        """log q in chunks; with an adaptive solver under the freeze guard (frozen points -> -inf)."""
        if not adaptive or s is screen:
            return torch.cat([s.log_prob(zc, hc if h is not None else None)
                              for zc, hc in zip(z.split(args.chunk), (z if h is None else h).split(args.chunk))])
        out, frozen = [], 0
        for zc, hc in zip(z.split(args.chunk), (z if h is None else h).split(args.chunk)):
            lq, fz, _ = frozen_log_prob(s, zc, None if h is None else hc, args.freeze_radius, args.freeze_cap)
            out.append(lq)
            frozen += int(fz.sum())
        if frozen:
            print(f"  freeze guard: {frozen} / {z.shape[0]} points frozen (density 0)", flush=True)
        return torch.cat(out)

    bank = sir.PathBank(args.bank, 12345, dev)
    g = torch.Generator(device=dev).manual_seed(0)
    perm = bank.valid[torch.randperm(bank.valid.shape[0], device=dev, generator=g)]
    idx = perm[: args.runs]
    z_c = bank.z[perm[args.runs: args.runs + args.L]]  # contrastive prior draws, disjoint from the runs
    z = bank.z[idx]
    with torch.no_grad():
        logp_true = log_prob(prior_sampler, z, None)
        logp_c = log_prob(prior_sampler, z_c, None)

    results, per_run = {}, {}
    for d in args.designs:
        name = d.split("=")[0]
        gd = torch.Generator(device=dev).manual_seed(1)
        with torch.no_grad():
            if d == "equal":
                tau = sir.equal_interval_taus(args.T).to(dev).expand(args.runs, -1)
            elif d == "random":
                tau = sir.random_taus(args.runs, args.T, gd, dev)
            elif d.split("=", 1)[1].startswith("file:"):
                saved = torch.load(d.split("=", 1)[1][5:], map_location=dev)
                tau = (saved["tau"] if isinstance(saved, dict) else saved)[: args.runs].float()
            else:
                pk = torch.load(d.split("=", 1)[1], map_location=dev)
                pol = Policy(enc, SPEC.eps_dim, pk["config"]["policy_hidden_dims"], "silu", None).to(dev)
                pol.net.load_state_dict(pk["policy"])
                tau, _, _ = rollout_designs(pol, enc, bank, idx, args.T)
            y = bank.observe(idx, tau)
            h = enc(y, sir.tau_to_eps(tau)[..., None])

            # InfoNCE: U(h_i, z) for the true z_i and the L shared contrastive z
            logq_true = log_prob(sampler, z, h)
            # the backward ODE can blow up for z far outside the posterior (log q -> -1e38 / nan); the critic maps
            # non-finite values to -1e4 (any critic gives a valid bound; for the true pair this can only lower it)
            u_true = torch.nan_to_num(logq_true - logp_true, nan=-1e4, posinf=-1e4, neginf=-1e4).clamp(min=-1e4)
            per = max(1, args.chunk // args.L)
            bounds = []
            for hs in h.split(per):
                n = hs.shape[0]
                lq = screen.log_prob(z_c.repeat(n, 1), hs.repeat_interleave(args.L, 0)).reshape(n, args.L)
                bounds.append(lq - logp_c[None])
            u_c = torch.cat(bounds)
            if adaptive:  # re-solve the pairs that matter with the guarded adaptive solver
                row_max = torch.maximum(torch.where(torch.isfinite(u_c), u_c, torch.tensor(-1e30, device=dev))
                                        .max(1).values, u_true.clamp(min=-1e30))
                keep = (torch.isfinite(u_c) & (u_c > row_max[:, None] - args.screen_margin)).reshape(-1)
                sel = keep.nonzero()[:, 0]
                print(f"{name}: screening kept {sel.numel()} / {keep.numel()} contrastive pairs", flush=True)
                lq_k = log_prob(sampler, z_c[sel % args.L], h[sel // args.L])
                u_c = u_c.reshape(-1)
                u_c[sel] = lq_k - logp_c[sel % args.L]
                u_c = u_c.reshape(args.runs, args.L)
            n_bad = (~torch.isfinite(u_c)).sum().item() + (u_c < -1e4).sum().item()
            u_c = torch.nan_to_num(u_c, nan=-1e4, posinf=-1e4, neginf=-1e4).clamp(min=-1e4)
            print(f"{d}: non-finite critic values: true pairs {(u_true <= -1e4).sum().item()} / {u_true.numel()}, "
                  f"contrastive {n_bad} / {u_c.numel()}", flush=True)
            u_all = torch.cat([u_true[:, None], u_c], 1)
            nce = u_true - torch.logsumexp(u_all, 1) + math.log(args.L + 1)
            ba = u_true

            # calibration: HPD coverage on the first --calib runs (rk4: forward dopri5 sampling can stall)
            cov = {}
            if args.calib > 0:
                lq_cal = logq_true if not adaptive else log_prob(screen, z, h)
                ranks = []
                for i0 in range(0, args.calib, max(1, args.chunk // args.calib_samples)):
                    hc = h[i0: min(args.calib, i0 + max(1, args.chunk // args.calib_samples))]
                    _, lq_s = screen.sample(args.calib_samples, hc, return_log_prob=True)
                    ranks.append((lq_s > lq_cal[i0: i0 + hc.shape[0], None]).float().mean(1))
                ranks = torch.cat(ranks)  # mass of the HPD region on whose boundary the true z lies
                cov = {f"{lv:.2f}": (ranks <= lv).float().mean().item() for lv in (0.5, 0.8, 0.95)}
        per_run[name] = {"infonce": nce.cpu(), "ba": ba.cpu(), "tau": tau.cpu()}
        se = lambda v: v.std().item() / math.sqrt(v.numel())
        results[name] = {"infonce": nce.mean().item(), "infonce_se": se(nce), "ba": ba.mean().item(),
                         "ba_se": se(ba), "logq": logq_true.mean().item(), "coverage": cov,
                         "tau_mean": tau.mean(0).tolist()}
        r = results[name]
        print(f"{name:14s} InfoNCE {r['infonce']:.3f} (s.e. {r['infonce_se']:.3f})  BA {r['ba']:.3f} "
              f"(s.e. {r['ba_se']:.3f})  coverage {cov}  mean taus {[round(t, 1) for t in r['tau_mean']]}", flush=True)
    if args.out_json is not None:
        args.out_json.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, **results}, indent=1))
        # per-run values (same runs, in the same order, for every design: paired differences)
        torch.save(per_run, args.out_json.with_suffix(".pt"))


if __name__ == "__main__":
    main()
