"""Design-independent quality of SIR posterior models (flows): which flow is the best posterior overall?

Metric: the held-out expected log-density of the true parameter, E[log q(z_true | h)], over a fixed, broad
distribution of histories h (not the designs of any policy). In expectation,
    E[-log q(z | h)] = H(z | h) + E[KL(p(z | h) || q(z | h))],
and the first term does not depend on the flow, so a gain in E[log q] is a gain of the same size in -E[KL]: a proper
score of the whole posterior (location, width, shape), in nats. (= the BA bound up to the constant E[log p(z)].)
History distribution: --runs held-out paths (bank seed 12345, never used in training), each scored under every
design family below and every history length q = 1..5 (the runs are split into 5 equal groups by q):
  uniform     q independent uniform times on (LB, UB);
  increasing  iDAD's increasing transform with u ~ U(-5, 5);
  grid        equally spaced, start U(LB, 50), spacing U(1, (UB - start) / 4), the first q;
  clustered   all q times uniform in a window of width U(1, 20) at a uniform position (in no training distribution).
Policy design sets (--policy-sets, a second table, not part of the overall score): NAME=POLICY.pt (our policy's
rollouts, with the encoder of the flow it was trained on, --policy-flow) or NAME=file:DESIGNS.pt (designs saved for the
held-out runs of eval_sir.py, e.g. iDAD's by idad_designs.py); scored on the same runs (the first min(--runs, saved)),
also at lengths 1-5 (the first q designs: the partial histories a policy reads during its rollout).
For a fixed set of designs, the paired difference in E[log q] between two flows is minus their difference in
E[KL(p(z | h) || q(z | h))]: all comparisons below are such paired differences.
The same paths, designs and lengths for every flow, so all differences are paired. log q: rk4 (--n-steps, default 50)
with the exact divergence. Coverage (secondary): fraction of the first --calib runs of each family whose true z lies
in the flow's 50 / 80 / 95% highest-density region (--calib-samples samples); ideal = the level.
Reports: overall mean log q (average over families and lengths), its paired difference to the first flow with the
standard error, and the breakdown by family and by length.
    eval_sir_posterior.py --flows REF.pt OTHER.pt OTHER.pt@ema ... [--out-json F]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from flow_bed.config import parse_args as parse_config_args
from flow_bed.models import PosteriorSampler
from flow_bed.models.policy import Policy
from flow_bed.simulators import sir
from flow_bed.sir import SPEC, load_flow, rollout_designs

FAMILIES = ["uniform", "increasing", "grid", "clustered"]
T = 5


def family_taus(family: str, b: int, q: int, gen: torch.Generator, dev) -> torch.Tensor:
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)
    if family == "uniform":
        return sir.LB + r(b, q) * (sir.UB - sir.LB)
    if family == "increasing":
        cols, prev = [], torch.full((b,), sir.LB, device=dev)
        for _ in range(q):
            prev = sir.next_tau(r(b) * 10 - 5, prev)
            cols.append(prev)
        return torch.stack(cols, 1)
    if family == "grid":
        start = sir.LB + r(b, 1) * (50.0 - sir.LB)
        sp = 1.0 + r(b, 1) * ((sir.UB - start) / (T - 1) - 1.0)
        return start + sp * torch.arange(q, device=dev)[None]
    if family == "clustered":
        w = 1.0 + r(b, 1) * 19.0
        lo = sir.LB + r(b, 1) * (sir.UB - sir.LB - w)
        return lo + r(b, q) * w
    raise ValueError(family)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flows", type=Path, nargs="+", required=True)
    p.add_argument("--runs", type=int, default=5000, help="held-out paths (split into 5 groups by history length)")
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--chunk", type=int, default=50_000)
    p.add_argument("--calib", type=int, default=200, help="coverage runs per design set (spread over the lengths)")
    p.add_argument("--calib-samples", type=int, default=500)
    p.add_argument("--policy-sets", nargs="*", default=[], help="NAME=POLICY.pt or NAME=file:DESIGNS.pt")
    p.add_argument("--policy-flow", type=Path, default=Path("sir_flow_s1.pt"))
    p.add_argument("--bank", type=int, default=50_000)
    p.add_argument("--out-json", type=Path, default=None)
    args = parse_config_args(p)
    dev = torch.device("cuda")

    bank = sir.PathBank(args.bank, 12345, dev)
    g = torch.Generator(device=dev).manual_seed(0)
    perm = bank.valid[torch.randperm(bank.valid.shape[0], device=dev, generator=g)]
    idx = perm[: args.runs]
    z = bank.z[idx]
    q_of = torch.arange(args.runs, device=dev) % T + 1  # history length per run, balanced
    # fixed designs: per family and length, for the runs of that length
    gd = torch.Generator(device=dev).manual_seed(1)
    designs = {f: {q: family_taus(f, int((q_of == q).sum()), q, gd, dev) for q in range(1, T + 1)} for f in FAMILIES}
    calib_mask = torch.zeros(args.runs, dtype=torch.bool, device=dev)
    calib_mask[: args.calib] = True
    # policy design sets: full T-step designs per run (n_runs may be smaller for saved designs)
    pol_designs = {}
    if args.policy_sets:
        _, pg, _ = load_flow(args.policy_flow, dev)
        for spec in args.policy_sets:
            name, src = spec.split("=", 1)
            with torch.no_grad():
                if src.startswith("file:"):
                    saved = torch.load(src[5:], map_location=dev)
                    tau = (saved["tau"] if isinstance(saved, dict) else saved)[: args.runs].float()
                else:
                    pk = torch.load(src, map_location=dev)
                    pol = Policy(pg.encoder, SPEC.eps_dim, pk["config"]["policy_hidden_dims"], "silu", None).to(dev)
                    pol.net.load_state_dict(pk["policy"])
                    tau, _, _ = rollout_designs(pol, pg.encoder, bank, idx, T)
            pol_designs[name] = tau
        del pg

    def score(sampler, enc, n, tau_of_q):
        """log q of the true z for the first n runs, each at its length q_of; coverage ranks of the calib runs."""
        lq = torch.empty(n, device=dev)
        ranks = []
        for q in range(1, T + 1):
            sel = (q_of[:n] == q).nonzero()[:, 0]
            tau = tau_of_q(q, sel)
            with torch.no_grad():
                y = bank.observe(idx[sel], tau)
                h = enc(y, sir.tau_to_eps(tau)[..., None])
                lq[sel] = torch.cat([sampler.log_prob(zc, hc) for zc, hc in
                                     zip(z[sel].split(args.chunk), h.split(args.chunk))])
                cs = calib_mask[sel].nonzero()[:, 0]
                step = max(1, args.chunk // args.calib_samples)
                for i0 in range(0, cs.numel(), step):
                    c = cs[i0: i0 + step]
                    _, lq_s = sampler.sample(args.calib_samples, h[c], return_log_prob=True)
                    ranks.append((lq_s > lq[sel][c, None]).float().mean(1))
        ranks = torch.cat(ranks)
        n_bad = int((~torch.isfinite(lq)).sum().item())
        cov = {f"{lv:.2f}": (ranks <= lv).float().mean().item() for lv in (0.5, 0.8, 0.95)}
        return torch.nan_to_num(lq, nan=-1e4, neginf=-1e4, posinf=-1e4), n_bad, cov

    logq_all, results = {}, {}
    for f_path in args.flows:
        prior, guidance, _ = load_flow(f_path, dev)
        enc = guidance.encoder
        sampler = PosteriorSampler(prior, guidance, solver="rk4", n_steps=args.n_steps, divergence="exact")
        res = {"family": {}, "length": {}, "policy": {}}
        per_fam, per_pol = {}, {}
        for fam in FAMILIES:
            per_fam[fam], n_bad, cov = score(sampler, enc, args.runs, lambda q, sel, fam=fam: designs[fam][q])
            res["family"][fam] = {"logq": per_fam[fam].mean().item(), "n_nonfinite": n_bad, "coverage": cov}
        for name, tau_full in pol_designs.items():
            n = tau_full.shape[0]
            per_pol[name], n_bad, cov = score(sampler, enc, n, lambda q, sel, t=tau_full: t[sel, :q])
            res["policy"][name] = {"logq": per_pol[name].mean().item(), "n_nonfinite": n_bad, "coverage": cov,
                                   "length": {q: per_pol[name][q_of[:n] == q].mean().item() for q in range(1, T + 1)}}
        overall = torch.stack([per_fam[f] for f in FAMILIES]).mean(0)  # per run: average over families
        logq_all[str(f_path)] = {"overall": overall, **per_fam, **{f"policy:{k}": v for k, v in per_pol.items()}}
        res["overall"] = overall.mean().item()
        res["overall_se"] = overall.std().item() / math.sqrt(args.runs)
        for q in range(1, T + 1):
            res["length"][q] = overall[q_of == q].mean().item()
        results[str(f_path)] = res
        print(f"scored {f_path}", flush=True)

    ref = str(args.flows[0])
    short = lambda s: s.split("/")[-1]

    def block(title: str, key: str, n: int, cov_of=None) -> None:
        """Per flow: the final score (all lengths) and the score for each length q: E[log q(z_true | h)], the paired
        difference to the reference flow and its standard error (difference = minus the difference in E[KL])."""
        qv = q_of[:n]
        masks = [("all q", torch.ones(n, dtype=torch.bool, device=dev))] + [(f"q={q}", qv == q) for q in range(1, T + 1)]
        print(f"\n=== {title}; paired differences to {short(ref)} (diff = -diff in E[KL]); cell: log q  diff +- s.e.")
        print(f"{'flow':30s}" + "".join(f"{m:>22s}" for m, _ in masks) + ("   coverage 50/80/95" if cov_of else ""))
        for f_path in args.flows:
            k = str(f_path)
            v, vr = logq_all[k][key], logq_all[ref][key]
            cells, out = [], {}
            for m, mask in masks:
                d = (v - vr)[mask]
                lq, dm, se = v[mask].mean().item(), d.mean().item(), d.std().item() / math.sqrt(d.numel())
                out[m] = {"logq": lq, "diff": dm, "diff_se": se}
                cells.append(f"{lq:8.3f} {dm:+.3f}+-{se:.3f}")
            results[k].setdefault("blocks", {})[key] = out
            line = f"{short(k):30s}" + "".join(f"{c:>22s}" for c in cells)
            if cov_of:
                c = cov_of(k)
                line += f"   {100 * c['0.50']:.0f}/{100 * c['0.80']:.0f}/{100 * c['0.95']:.0f}"
            print(line)

    block(f"RANKING: average over the {len(FAMILIES)} design families ({', '.join(FAMILIES)}), {args.runs} held-out runs",
          "overall", args.runs)
    for fam in FAMILIES:
        block(f"design family {fam}, {args.runs} runs", fam, args.runs,
              cov_of=lambda k, fam=fam: results[k]["family"][fam]["coverage"])
    for name in pol_designs:
        block(f"policy designs {name} (partial histories), {pol_designs[name].shape[0]} runs", f"policy:{name}",
              pol_designs[name].shape[0], cov_of=lambda k, name=name: results[k]["policy"][name]["coverage"])
    bad = {short(str(f)): [results[str(f)]["family"][fm]["n_nonfinite"] for fm in FAMILIES] for f in args.flows}
    if any(any(v) for v in bad.values()):
        print("\nnon-finite log q per family (set to -1e4):", bad)
    if args.out_json is not None:
        args.out_json.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, **results}, indent=1))
        torch.save({k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in logq_all.items()},
                   args.out_json.with_suffix(".pt"))


if __name__ == "__main__":
    main()
