"""SIR stage 2: decoupled design policy on the frozen flow of train_sir_flow.py, divergence utility, as
runs/scripts/training/train_lf_policy.py (one tau per batch, one theta_0 per sample, shared by all steps; q ~ U{q_min,
..., q_max}; one-cycle max lr 1e-4, 2% warm-up; grad clip 5), with:
  - designs: increasing measurement times tau_t = tau_{t-1} + sigmoid(u_t) (UB - tau_{t-1}) (iDAD), u_t = policy head
    on the flow's encoded history (head only is trained);
  - observations from a per-process path bank (re-simulated every --refresh steps), d y / d tau = central difference;
  - exact divergence (theta is 2-D).
--policy-term sum | final_random, --detach-policy-input as in the LF script. --max-steps stops early (smoke tests).
--dy-window H: d y / d tau = central difference over +-H time units of the path (default 0.01 = one grid step, iDAD's
gradient; +-0.5 removes most of the jitter of the simulator gradient and trains markedly better policies).
--flow-time sample | batch: the flow time of the loss, one per path (default since 2026-11-07; removes the
shared-flow-time part of the batch-gradient noise, 0-30% at +-0.01, 20-40% at +-0.5, at no cost) or one shared by
the whole batch (all policies trained before 2026-11-07; their configs have no flow_time key), or strat8 (2026-11-08):
8 stratified flow times (k + U_k) / 8 per path, one rollout per path, the loss = the mean over all 8 x batch terms
(the CSV 'tau' column then holds the first path's first time, in [0, 1/8)).
--init-frac F: the policy's output bias starts at logit(F), so the untrained policy moves each design a fraction F of
the remaining window (F = 0.15: ~15, 28, 39, 48, 56) instead of 0.5 (~50, 75, 87, 94, 97, after most epidemics).
--config file.yaml sets the defaults of these arguments (see runs/configs/sir and flow_bed.config).
    python runs/scripts/training/train_sir_policy.py --config runs/configs/sir/policy_finalq.yaml --out POLICY.pt
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from flow_bed.config import parse_args
from flow_bed.lf.training import q_max_at
from flow_bed.models.policy import Policy
from flow_bed.models.posterior_sampler import exact_divergence
from flow_bed.simulators import sir
from flow_bed.sir import SPEC, load_flow, rollout_designs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt", type=Path, default=None, help="required, as a flag or in the config")
    p.add_argument("--policy-term", choices=["sum", "final_random"], default="sum")
    p.add_argument("--detach-policy-input", action="store_true")
    p.add_argument("--Q", type=int, default=5)
    p.add_argument("--q-min", type=int, default=2)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--steps-per-epoch", type=int, default=781)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--policy-hidden-dims", type=int, nargs="+", default=[128, 128, 128, 128])
    p.add_argument("--bank", type=int, default=25_000)
    p.add_argument("--refresh", type=int, default=2_500)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log-csv", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dy-window", type=float, default=0.01, help="half-width (time units) of d y / d tau")
    p.add_argument("--init-frac", type=float, default=None, help="initial sigmoid(u) of every design step")
    p.add_argument("--flow-time", choices=["sample", "batch", "strat8"], default="sample",
                   help="flow time of the loss: one per path (default), one per batch (before 2026-11-07), or 8 "
                        "stratified per path (one rollout, the divergence averaged over the 8 times)")
    args = parse_args(p)
    if args.flow_ckpt is None:
        p.error("--flow-ckpt is required (as a flag or in the config)")
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    cfg = {**vars(args), "flow_ckpt": str(args.flow_ckpt), "out": str(args.out), "log_csv": str(args.log_csv),
           "model": "sir"}

    prior, guidance, _ = load_flow(args.flow_ckpt, dev)
    enc = guidance.encoder
    policy = Policy(enc, SPEC.eps_dim, args.policy_hidden_dims, "silu", None).to(dev)
    if args.init_frac is not None:
        with torch.no_grad():
            policy.net[-1].bias.fill_(torch.logit(torch.tensor(args.init_frac)).item())
    params = list(policy.net.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, div_factor=20.0, final_div_factor=20.0,
                                                total_steps=total, pct_start=0.02)
    warmup = int(0.002 * total)
    bank = sir.PathBank(args.bank, args.seed, dev)
    w = max(1, round(args.dy_window / bank.dt_grid))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = log_file = None
    if args.log_csv is not None:
        log_file = open(args.log_csv, "w", newline="")  # noqa: SIM115
        writer = csv.writer(log_file)
        writer.writerow(["step", "epoch", "q", "loss", "lr", "seconds", "tau"])

    def save(epoch):
        torch.save({"config": cfg, "policy": policy.net.state_dict(), "epoch": epoch}, args.out)

    t0 = time.time()
    steps = total if args.max_steps is None else min(total, args.max_steps)
    for step in range(steps):
        if step > 0 and step % args.refresh == 0:
            bank.refresh()
        q = int(torch.randint(args.q_min, q_max_at(step, args.q_min, args.Q, warmup) + 1, ()))
        idx = bank.sample_idx(args.batch)
        K = 8 if args.flow_time == "strat8" else 1  # flow times per path (the rollout is shared by them)
        z = bank.z[idx].repeat_interleave(K, 0)
        if args.flow_time == "strat8":  # t_k = (k + U_k) / 8, k = 0..7, per path
            tau = ((torch.arange(K, device=dev) + torch.rand(args.batch, K, device=dev)) / K).reshape(-1)
        elif args.flow_time == "sample":
            tau = torch.rand(args.batch, device=dev)
        else:
            tau = torch.rand(1, device=dev).expand(args.batch)
        x = (tau[:, None] * z + prior.lambda_t(tau)[:, None] * torch.randn_like(z)).requires_grad_(True)
        _, _, hs = rollout_designs(policy, enc, bank, idx, q, args.detach_policy_input, w)
        terms = hs if args.policy_term == "sum" else hs[-1:]
        loss = sum(exact_divergence(guidance(x, tau, h.repeat_interleave(K, 0)), x, create_graph=True).mean()
                   for h in terms)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        lr = opt.param_groups[0]["lr"]
        opt.step()
        sched.step()
        if writer is not None:
            writer.writerow([step, step // args.steps_per_epoch, q, f"{loss.item():.6g}", f"{lr:.6g}",
                             f"{time.time() - t0:.1f}", f"{tau[0].item():.4f}"])
        if (step + 1) % args.steps_per_epoch == 0:
            epoch = (step + 1) // args.steps_per_epoch
            if log_file is not None:
                log_file.flush()
            if epoch % args.ckpt_every == 0:
                save(epoch)
            mem = torch.cuda.max_memory_allocated(dev) / 2**30
            print(f"[sir-policy] epoch {epoch:4d}  q {q}  loss {loss.item():.4f}  peak mem {mem:.2f} GB  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if args.max_steps is not None:
        torch.cuda.synchronize()
        el = time.time() - t0
        print(f"SMOKE {steps} steps in {el:.1f}s = {1000 * el / steps:.1f} ms/step; full {total} steps -> "
              f"{el / steps * total / 3600:.1f} h; peak mem {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GB")
    save(steps // args.steps_per_epoch)
    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    main()
