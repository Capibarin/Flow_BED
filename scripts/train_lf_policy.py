"""Stage 2 for location finding: train the design policy on top of the frozen flow from scripts/train_lf_flow.py.

The policy shares the guidance model's (frozen) history encoder and only its head is trained. Settings follow JADAI's
joint phase: 400 epochs x 781 steps, Adam with a one-cycle schedule (max lr 1e-4), gradient clipping at 5, and
q ~ U{q_min, ..., q_max} adaptive steps per batch. Both utilities target EIG = E[log p(theta | h)] - E[log p(theta)]:

--utility divergence  EIG = -E[div_x w(x, tau, h)] at x = tau * theta + lambda_tau * theta_0, tau ~ U[0, 1]
                      (the posterior flow contracts more than the prior flow). As in JADAI's rollout utility, one tau
                      per batch and one theta_0 per sample are shared by all steps, and the policy minimises
                      sum_{t=1..q} div w(x, tau, h_{1:t}) (exact divergence by default; with Hutchinson, one
                      probe per sample, also shared by all steps).
--utility density     E[log q(theta | h_{1:q})] at the end of the rollout (Barber-Agakov lower bound on EIG), with
                      log q from the guided flow ODE (dopri5 by default, Hutchinson divergence by default, autograd).

    python scripts/train_lf_policy.py --utility divergence --out checkpoints/lf_policy_div.pt
    python scripts/train_lf_policy.py --utility density --batch 64 --out checkpoints/lf_policy_density.pt
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from train_lf_flow import build_flow, q_max_at

from flow_bed.models import PosteriorSampler
from flow_bed.models.policy import Policy
from flow_bed.models.posterior_sampler import exact_divergence, hutchinson_divergence
from flow_bed.simulators.location_finding import LocationFinding


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt", type=Path, default=Path("checkpoints/lf_flow.pt"))
    p.add_argument("--utility", choices=["divergence", "density"], default="divergence")
    p.add_argument("--Q", type=int, default=30, help="number of adaptive steps (time horizon)")
    p.add_argument("--q-min", type=int, default=2)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--steps-per-epoch", type=int, default=781)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--policy-hidden-dims", type=int, nargs="+", default=[128, 128, 128, 128])
    p.add_argument("--divergence", choices=["exact", "hutchinson"], default=None,
                   help="divergence estimator (default: exact for --utility divergence, hutchinson for density)")
    p.add_argument("--solver", default="dopri5", help="ODE solver for --utility density")
    p.add_argument("--tol", type=float, default=1e-3, help="atol = rtol of adaptive solvers for --utility density")
    p.add_argument("--n-steps", type=int, default=20, help="steps of fixed-step solvers for --utility density")
    p.add_argument("--out", type=Path, default=Path("checkpoints/lf_policy.pt"))
    p.add_argument("--log-csv", type=Path, default=None, help="write one row per training step (for convergence plots)")
    p.add_argument("--detach-policy-input", action="store_true",
                   help="detach the encoded history the policy sees (as JADAI): no gradient through its input")
    p.add_argument("--ckpt-every", type=int, default=10, help="also save the policy to --out every this many epochs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.divergence is None:
        args.divergence = "exact" if args.utility == "divergence" else "hutchinson"
    cfg = {**vars(args), "flow_ckpt": str(args.flow_ckpt), "out": str(args.out), "log_csv": str(args.log_csv)}

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    sim = LocationFinding()

    # frozen flow: prior, guidance and history encoder
    ck = torch.load(args.flow_ckpt, map_location=dev)
    prior, guidance = build_flow(ck["config"], sim)
    prior.load_state_dict(ck["prior"])
    guidance.load_state_dict(ck["guidance"])
    guidance.to(dev).eval().requires_grad_(False)
    sampler = PosteriorSampler(
        prior, guidance, solver=args.solver, n_steps=args.n_steps, atol=args.tol, rtol=args.tol, divergence=args.divergence
    )
    nfe = [0]  # number of vector field evaluations of the density ODE, for monitoring
    sampler.cnf.register_forward_hook(lambda *_: nfe.__setitem__(0, nfe[0] + 1))

    policy = Policy(guidance.encoder, sim.eps_dim, args.policy_hidden_dims).to(dev)
    params = list(policy.net.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, div_factor=20.0, final_div_factor=20.0, total_steps=total, pct_start=0.02
    )
    warmup = int(0.002 * total)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_file = writer = None
    if args.log_csv is not None:
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log_csv, "w", newline="")  # noqa: SIM115 (closed after training)
        writer = csv.writer(log_file)
        writer.writerow(["step", "epoch", "q", "loss", "loss_per_q", "lr", "nfe", "seconds"])

    t0 = time.time()
    for step in range(total):
        q = int(torch.randint(args.q_min, q_max_at(step, args.q_min, args.Q, warmup) + 1, ()))
        theta = sim.sample_prior(args.batch, dev)
        eps = torch.zeros(args.batch, 0, sim.eps_dim, device=dev)
        y = torch.zeros(args.batch, 0, sim.y_dim, device=dev)
        if args.utility == "divergence":
            # one tau per batch and one theta_0 per sample, shared by all steps of the rollout
            tau = torch.rand(1, device=dev).expand(args.batch)
            lam = prior.lambda_t(tau)[:, None]
            x = (tau[:, None] * theta + lam * torch.randn_like(theta)).requires_grad_(True)
            probe = torch.randn_like(theta)  # Hutchinson probe, shared by all steps
        loss = 0.0
        for _ in range(q):
            if args.detach_policy_input:
                xi = policy.net(guidance.encoder(y, eps).detach())[:, None]
            else:
                xi = policy(y, eps)[:, None]  # next design (B, 1, dx), from the history so far
            eps = torch.cat([eps, xi], dim=1)
            y = torch.cat([y, sim.simulate(theta, xi)], dim=1)  # reparametrised: gradients reach the design
            if args.utility == "divergence":
                w = guidance(x, tau, guidance.encoder(y, eps))
                if args.divergence == "exact":
                    div = exact_divergence(w, x, create_graph=True)
                else:
                    div = hutchinson_divergence(w, x, probe, create_graph=True)
                loss = loss + div.mean()
        if args.utility == "density":
            nfe[0] = 0
            loss = -sampler.log_prob(theta, guidance.encoder(y, eps)).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        lr = opt.param_groups[0]["lr"]
        opt.step()
        sched.step()
        if writer is not None:
            nfe_step = nfe[0] if args.utility == "density" else ""
            writer.writerow([step, step // args.steps_per_epoch, q, f"{loss.item():.6g}", f"{loss.item() / q:.6g}",
                             f"{lr:.6g}", nfe_step, f"{time.time() - t0:.1f}"])
        if (step + 1) % args.steps_per_epoch == 0:
            epoch = (step + 1) // args.steps_per_epoch
            if log_file is not None:
                log_file.flush()
            if epoch % args.ckpt_every == 0:
                torch.save({"config": cfg, "policy": policy.net.state_dict(), "epoch": epoch}, args.out)
            extra = f"  nfe {nfe[0]}" if args.utility == "density" else ""
            mem = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0
            print(f"[policy-{args.utility}] epoch {epoch:4d}  q {q:2d}  loss {loss.item():.4f}{extra}  "
                  f"peak mem {mem:.2f} GB  ({time.time() - t0:.0f}s)", flush=True)

    torch.save({"config": cfg, "policy": policy.net.state_dict(), "epoch": args.epochs}, args.out)
    if log_file is not None:
        log_file.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
