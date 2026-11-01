"""Per-tau RMS of the design gradient of the divergence utility, the input of the variance-optimal tau density.

Frozen posterior flow (h-parametrisation, w = c_tau h with c_tau = 1 - tau). For each tau on a grid (16 points
k / 15, refined by 8 points in [0.95, 0.999], where the RMS peaks) and each sample (theta ~ p(theta), its own design
sequence xi_{1:n}, y ~ p(y | theta, xi), theta_0, probe):
    u = -div w(x, tau, h(y, xi)),   x = tau theta + lambda_tau theta_0,   g = d u / d xi_{1:n}  (per sample)
    m2(tau) = E ||g||^2,   rho(tau) ∝ sqrt(m2(tau))   (piecewise linear on the grid; see figures/plot_rho.py)
div: one-probe Hutchinson (as in training) and exact. Designs: "random" xi ~ N(0, I), and "policy" = rollouts of
--policy (designs then held fixed, y recomputed with the same noise). History lengths: n = 1, 10, 30 and "mix" =
q ~ U{1..30} (every q equally often, the final_random training mixture). Writes --out (JSON).

    python runs/scripts/analysis/tau_rms.py --flow-ckpt checkpoints/lf_flow_attn.pt \
        --policy checkpoints/lf_policy_div_finalrandq.pt --out checkpoints/tau_rms.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flow_bed.lf.builders import build_flow, build_sim, load_policy
from flow_bed.models.posterior_sampler import exact_divergence, hutchinson_divergence


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True, help="policy checkpoint for the 'policy' designs")
    p.add_argument("--out", type=Path, required=True, help="output JSON")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--batches", type=int, default=8, help="batches per (designs, n) for n = 1, 10, 30")
    p.add_argument("--batches-mix", type=int, default=60, help="batches for the q ~ U{1..30} mixture")
    p.add_argument("--quick", action="store_true", help="2 batches each (a smoke test)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.quick:
        args.batches = args.batches_mix = 2
    torch.set_num_threads(args.threads)

    dev = torch.device(args.device)
    ck = torch.load(args.flow_ckpt, map_location=dev)
    sim = build_sim(ck["config"])
    prior, guidance = build_flow(ck["config"], sim)
    prior.load_state_dict(ck["prior"])
    guidance.load_state_dict(ck["guidance"])
    prior.to(dev).eval().requires_grad_(False)
    guidance.to(dev).eval().requires_grad_(False)
    policy = load_policy(torch.load(args.policy, map_location=dev), guidance, sim).to(dev).eval().requires_grad_(False)

    taus = torch.cat([torch.linspace(0, 1, 16), torch.tensor([0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999])])
    taus = taus.sort()[0]
    B, M = args.batch, args.batches
    ns = {"1": [1] * M, "10": [10] * M, "30": [30] * M, "mix": [1 + i % 30 for i in range(args.batches_mix)]}

    @torch.no_grad()
    def designs(kind, theta, n):
        """(xi (B, n, dx), observation noise (B, n, 1)) from random designs or a policy rollout."""
        noise = torch.randn(theta.shape[0], n, 1, device=dev)
        if kind == "random":
            return sim.sample_design(theta.shape[0], n, dev), noise
        xi = torch.zeros(theta.shape[0], 0, sim.dx, device=dev)
        y = torch.zeros(theta.shape[0], 0, 1, device=dev)
        for t in range(n):
            x_t = policy(y, xi)[:, None]
            xi = torch.cat([xi, x_t], 1)
            y = torch.cat([y, sim.log_mu(theta, x_t) + sim.sigma * noise[:, t:t + 1]], 1)
        return xi, noise

    results, t0 = {}, time.time()
    for kind in ["random", "policy"]:
        for nkey, n_list in ns.items():
            g2 = {d: np.zeros((len(n_list), len(taus))) for d in ["hu", "ex"]}  # per batch: mean ||g||^2
            for b, n in enumerate(n_list):
                torch.manual_seed(100_003 * b + 7919 * n + (kind == "policy"))
                theta = sim.sample_prior(B, dev)
                xi0, noise = designs(kind, theta, n)
                xi = xi0.clone().requires_grad_(True)
                y = sim.log_mu(theta, xi) + sim.sigma * noise
                hist = guidance.encoder(y, xi)
                for k, tau_k in enumerate(taus.tolist()):
                    tau = torch.full((B,), tau_k, device=dev)
                    x = (tau[:, None] * theta + prior.lambda_t(tau)[:, None] * torch.randn_like(theta)).requires_grad_(True)
                    w = guidance(x, tau, hist)
                    for d, div in [("hu", hutchinson_divergence(w, x, torch.randn_like(theta), True)),
                                   ("ex", exact_divergence(w, x, True))]:
                        (g,) = torch.autograd.grad(-div.sum(), xi, retain_graph=True)
                        g2[d][b, k] = (g.reshape(B, -1) ** 2).sum(-1).mean().item()
            res = {}
            for d in ["hu", "ex"]:
                m2, se = g2[d].mean(0), g2[d].std(0) / np.sqrt(len(n_list))
                res[d] = {"m2": m2.tolist(), "m2_se": se.tolist(), "rms": np.sqrt(m2).tolist()}
            results[f"{kind}_{nkey}"] = res
            print(f"[{kind} n={nkey}] ({time.time() - t0:.0f}s)  rms hu: "
                  + " ".join(f"{v:.2f}" for v in res["hu"]["rms"]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"flow_ckpt": str(args.flow_ckpt), "policy": str(args.policy), "batch": B, "batches": M,
            "batches_mix": args.batches_mix}
    with open(args.out, "w") as f:
        json.dump({"taus": taus.tolist(), "meta": meta, "results": results}, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
