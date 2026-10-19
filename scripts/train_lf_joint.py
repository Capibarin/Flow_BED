"""Location finding with JADAI's training schedule: pre-training on random designs, then policy and guidance jointly.

Phase 1 (pre-training, random designs): prior flow, then history encoder + guidance; --pre-epochs x 781 steps each,
    one-cycle schedule (max lr 1e-3, div 1e5, final div 2, 8% warm-up).
Phase 2 (joint): --joint-epochs x 781 steps, one-cycle (max lr 1e-4, div 20, final div 20, 2% warm-up) for both
    parts, gradient clipping at 5 per part. Per batch one rollout of q ~ U{q_min, ..., q_max} designs from the policy;
    as in JADAI, the noise is shared by all steps of a rollout (flow times tau_i = (u_0 + i / B) mod 1 and theta_0
    per sample) and the policy sees the detached history state (unless --no-detach-policy-input).
      policy   : minimises sum_t div w(x, tau, h_{1:t}) at x = tau theta + lambda_tau theta_0 (Hutchinson, -EIG)
      guidance : minimises sum_t of its score loss on the policy's histories h_{1:t} (designs detached),
                 plus --random-mix x q x its score loss on a batch of q random designs (default 0, as JADAI)
    The prior stays frozen after phase 1. Unlike JADAI, q_min defaults to 1.

Writes <out>_flow.pt (prior + guidance, loadable by build_flow) and <out>_policy.pt, also every --ckpt-every epochs.

    python scripts/train_lf_joint.py --out checkpoints/lf_joint --log-csv /vol/main/checkpoints/logs/lf_joint.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from train_lf_flow import build_guidance, q_max_at, train

from flow_bed.models import LikelihoodGuidanceModel, PriorFlowModel
from flow_bed.models.policy import Policy
from flow_bed.models.posterior_sampler import hutchinson_divergence
from flow_bed.simulators.location_finding import LocationFinding


def stratified_times(batch: int, device) -> Tensor:
    """JADAI's low-discrepancy flow times t_i = (u_0 + i / B) mod 1, shape (B,)."""
    return (torch.rand(1, device=device) + torch.arange(batch, device=device) / batch) % 1


def score_loss(guidance: LikelihoodGuidanceModel, theta1, theta0, t, eps, y) -> Tensor:
    """The guidance's score loss (as in shared_step) with given flow times t (B,) and base noise theta0."""
    prior = guidance.prior
    t_ = t[:, None]
    lam, lam_sp = prior.lambda_t(t_), prior.lambda_sp_t(t_)
    theta_t = t_ * theta1 + lam * theta0
    u_t = theta1 + lam_sp / lam * theta0
    c_t = lam**2 - lam_sp * t_
    with torch.no_grad():
        _, s_prior = prior(theta_t, t)
    w = guidance(theta_t, t, guidance.encoder(y, eps))
    return F.mse_loss(c_t * s_prior + t_ * w, t_ * u_t - theta_t)


def save(cfg, prior, guidance, policy, out: str, epoch: int) -> None:
    torch.save({"config": cfg, "prior": prior.state_dict(), "guidance": guidance.state_dict(), "epoch": epoch},
               f"{out}_flow.pt")
    torch.save({"config": cfg, "policy": policy.net.state_dict(), "epoch": epoch}, f"{out}_policy.pt")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--Q", type=int, default=30)
    p.add_argument("--q-min", type=int, default=1)
    p.add_argument("--pre-epochs", type=int, default=50)
    p.add_argument("--joint-epochs", type=int, default=400)
    p.add_argument("--steps-per-epoch", type=int, default=781)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr-pre", type=float, default=1e-3)
    p.add_argument("--lr-joint", type=float, default=1e-4)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512], help="prior network")
    p.add_argument("--guidance-hidden-dims", type=int, nargs="+", default=[512, 512, 512, 512])
    p.add_argument("--param", choices=["w", "h"], default="h")
    p.add_argument("--enc-hidden-dims", type=int, nargs="+", default=[128, 128])
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--policy-hidden-dims", type=int, nargs="+", default=[128, 128, 128, 128])
    p.add_argument("--no-detach-policy-input", action="store_true",
                   help="let the policy loss backpropagate through the history state to earlier designs")
    p.add_argument("--random-mix", type=float, default=0.0,
                   help="weight of the guidance's score loss on random designs during joint training (x q)")
    p.add_argument("--out", default="checkpoints/lf_joint", help="prefix of <out>_flow.pt and <out>_policy.pt")
    p.add_argument("--log-csv", type=Path, default=None, help="one row per joint training step")
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = {**vars(args), "log_csv": str(args.log_csv), "encoder": "deepsets", "attn_layers": 4, "attn_heads": 2}

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    sim = LocationFinding()
    B, spe = args.batch, args.steps_per_epoch

    # ---- phase 1: pre-training on random designs
    pre_total = args.pre_epochs * spe
    pre_warmup = int(0.002 * pre_total)
    prior = PriorFlowModel(sim.dim, hidden_dims=args.hidden_dims).to(dev)
    train("prior", list(prior.parameters()), lambda step: prior.shared_step(sim.sample_prior(B, dev)),
          pre_total, args.lr_pre, spe)
    prior.eval()
    guidance = build_guidance(cfg, sim, prior).to(dev)

    def pre_loss(step):
        q = int(torch.randint(args.q_min, q_max_at(step, args.q_min, args.Q, pre_warmup) + 1, ()))
        return guidance.shared_step(sim.sample_joint(B, q, dev))

    g_params = [p for p in guidance.parameters() if p.requires_grad]  # encoder + guidance net (prior is frozen)
    train("guidance-pre", g_params, pre_loss, pre_total, args.lr_pre, spe)

    # ---- phase 2: joint training
    policy = Policy(guidance.encoder, sim.eps_dim, args.policy_hidden_dims).to(dev)
    p_params = list(policy.net.parameters())
    total = args.joint_epochs * spe
    warmup = int(0.002 * total)

    def one_cycle(params):
        opt = torch.optim.Adam(params, lr=args.lr_joint)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr_joint, div_factor=20.0, final_div_factor=20.0, total_steps=total, pct_start=0.02
        )
        return opt, sched

    opt_p, sched_p = one_cycle(p_params)
    opt_g, sched_g = one_cycle(g_params)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    log_file = writer = None
    if args.log_csv is not None:
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log_csv, "w", newline="")  # noqa: SIM115 (closed after training)
        writer = csv.writer(log_file)
        writer.writerow(["step", "epoch", "q", "policy_loss", "guidance_loss", "lr", "seconds"])

    t0 = time.time()
    for step in range(total):
        q = int(torch.randint(args.q_min, q_max_at(step, args.q_min, args.Q, warmup) + 1, ()))
        theta = sim.sample_prior(B, dev)
        # noise shared by all steps of the rollout
        tau_p = stratified_times(B, dev)
        x = (tau_p[:, None] * theta + prior.lambda_t(tau_p)[:, None] * torch.randn_like(theta)).requires_grad_(True)
        probe = torch.randn_like(theta)
        tau_g, theta0_g = stratified_times(B, dev), torch.randn_like(theta)

        eps = torch.zeros(B, 0, sim.eps_dim, device=dev)
        y = torch.zeros(B, 0, sim.y_dim, device=dev)
        policy_loss = guidance_loss = 0.0
        for _ in range(q):
            h = guidance.encoder(y, eps)
            xi = policy.net(h if args.no_detach_policy_input else h.detach())[:, None]  # JADAI detaches
            eps = torch.cat([eps, xi], dim=1)
            y = torch.cat([y, sim.simulate(theta, xi)], dim=1)
            w = guidance(x, tau_p, guidance.encoder(y, eps))
            policy_loss = policy_loss + hutchinson_divergence(w, x, probe, create_graph=True).mean()
            guidance_loss = guidance_loss + score_loss(guidance, theta, theta0_g, tau_g, eps.detach(), y.detach())
        if args.random_mix > 0:  # keep the guidance accurate off the policy's histories
            guidance_loss = guidance_loss + args.random_mix * q * guidance.shared_step(sim.sample_joint(B, q, dev))

        # policy: gradient of the divergence utility w.r.t. the policy weights only
        grads = torch.autograd.grad(policy_loss, p_params)
        for prm, g in zip(p_params, grads):
            prm.grad = g
        torch.nn.utils.clip_grad_norm_(p_params, max_norm=5.0)
        lr = opt_p.param_groups[0]["lr"]
        opt_p.step()
        sched_p.step()
        # guidance + encoder: score loss on the policy's (detached) histories
        opt_g.zero_grad()
        guidance_loss.backward()
        torch.nn.utils.clip_grad_norm_(g_params, max_norm=5.0)
        opt_g.step()
        sched_g.step()

        if writer is not None:
            writer.writerow([step, step // spe, q, f"{policy_loss.item():.6g}", f"{guidance_loss.item():.6g}",
                             f"{lr:.6g}", f"{time.time() - t0:.1f}"])
        if (step + 1) % spe == 0:
            epoch = (step + 1) // spe
            if log_file is not None:
                log_file.flush()
            print(f"[joint] epoch {epoch:4d}  q {q:2d}  policy {policy_loss.item():.4f}  "
                  f"guidance {guidance_loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
            if epoch % args.ckpt_every == 0:
                save(cfg, prior, guidance, policy, args.out, epoch)

    save(cfg, prior, guidance, policy, args.out, args.joint_epochs)
    if log_file is not None:
        log_file.close()
    print(f"saved {args.out}_flow.pt and {args.out}_policy.pt")


if __name__ == "__main__":
    main()
