"""Stage 2 for location finding: train the design policy on top of the frozen flow from scripts/train_lf_flow.py.

The policy shares the guidance model's (frozen) history encoder and only its head is trained; with --own-encoder it has
a separate, freshly initialised history encoder that is trained with the head. Settings follow JADAI's
joint phase: 400 epochs x 781 steps, Adam with a one-cycle schedule (max lr 1e-4), gradient clipping at 5, and
q ~ U{q_min, ..., q_max} adaptive steps per batch. Both utilities target EIG = E[log p(theta | h)] - E[log p(theta)]:

--utility divergence  EIG = -E[div_x w(x, tau, h)] at x = tau * theta + lambda_tau * theta_0, tau ~ U[0, 1]
                      (the posterior flow contracts more than the prior flow). As in JADAI's rollout utility, one tau
                      per batch and one theta_0 per sample are shared by all steps, and the policy minimises
                      sum_{t=1..q} div w(x, tau, h_{1:t}) (exact divergence by default; with Hutchinson, one
                      probe per sample, also shared by all steps).
--utility density     E[log q(theta | h_{1:q})] at the end of the rollout (Barber-Agakov lower bound on EIG), with
                      log q from the guided flow ODE (dopri5 by default, Hutchinson divergence by default, autograd).
--policy-term final   (divergence utility) the final term div w(x, tau, h_{1:q}) alone, with q = q_max (= Q after the
                      warm-up) instead of a random q: -EIG of the whole rollout. The sum counts the information of
                      experiment s (q - s + 1) times and favours early information.
--policy-term final_random  the final term alone, like final, but with q ~ U{q_min, ..., q_max} as for sum: experiment
                      s counts once when q >= s, so the weights fall linearly instead of quadratically.
--tau-max             (divergence utility) tau ~ U[0, tau_max]: the information about theta blurred by noise of std
                      (1 - tau_max) / tau_max, which caps the reward for resolving a source beyond that scale;
                      --tau-anneal-frac f raises tau_max linearly to 1 over the last f of the steps (the policy at the
                      start of the annealing is also saved as <out>_before_anneal.pt).
--config file.yaml   sets the defaults of these arguments (see configs/lf and flow_bed.config).

    python scripts/train_lf_policy.py --config configs/lf/policy_div.yaml --out checkpoints/lf_policy_div.pt
    python scripts/train_lf_policy.py --utility divergence --out checkpoints/lf_policy_div.pt
    python scripts/train_lf_policy.py --utility density --batch 64 --out checkpoints/lf_policy_density.pt
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import torch
from train_lf_flow import build_encoder, build_flow, q_max_at

from flow_bed.config import parse_args
from flow_bed.models import LikelihoodGuidanceModel, PosteriorSampler
from flow_bed.models.flow_matching import LEGACY_ACTIVATION
from flow_bed.models.policy import Policy
from flow_bed.models.posterior_sampler import exact_divergence, hutchinson_divergence
from flow_bed.simulators.location_finding import LocationFinding

ENCODER_KEYS = ("encoder", "enc_hidden_dims", "embed_dim", "attn_layers", "attn_heads", "y_loc", "y_scale")


def load_policy(pk: dict, guidance: LikelihoodGuidanceModel, sim: LocationFinding) -> Policy:
    """Policy from a checkpoint of this script: on its own encoder if it has one, else on the guidance model's."""
    if "encoder" in pk:
        encoder = build_encoder(pk["config"]["policy_encoder"], sim)
        encoder.load_state_dict(pk["encoder"])
    else:
        encoder = guidance.encoder
    policy = Policy(encoder, sim.eps_dim, pk["config"]["policy_hidden_dims"],
                    pk["config"].get("activation", LEGACY_ACTIVATION), pk["config"].get("design_bound"),
                    pk["config"].get("bound_type", "radial"))
    policy.net.load_state_dict(pk["policy"])
    return policy


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
    p.add_argument("--activation", choices=["silu", "selu"], default="silu",
                   help="activation of the policy head (and of its own encoder with --own-encoder)")
    p.add_argument("--design-bound", type=float, default=None,
                   help="bound s on the designs (default: unbounded); see flow_bed.models.policy")
    p.add_argument("--bound-type", choices=["radial", "coord"], default="radial",
                   help="radial: |eps| < s; coord: each coordinate in (-s, s)")
    p.add_argument("--divergence", choices=["exact", "hutchinson"], default=None,
                   help="divergence estimator (default: exact for --utility divergence, hutchinson for density)")
    p.add_argument("--solver", default="dopri5", help="ODE solver for --utility density")
    p.add_argument("--tol", type=float, default=1e-3, help="atol = rtol of adaptive solvers for --utility density")
    p.add_argument("--n-steps", type=int, default=20, help="steps of fixed-step solvers for --utility density")
    p.add_argument("--out", type=Path, default=Path("checkpoints/lf_policy.pt"))
    p.add_argument("--log-csv", type=Path, default=None, help="write one row per training step (for convergence plots)")
    p.add_argument("--detach-policy-input", action="store_true",
                   help="detach the encoded history the policy sees (as JADAI): no gradient through its input")
    p.add_argument("--own-encoder", action="store_true",
                   help="give the policy its own history encoder (fresh, trained with the head) instead of the flow's")
    p.add_argument("--policy-enc-hidden-dims", type=int, nargs="+", default=None,
                   help="hidden dims of the policy's own encoder (default: as the flow's encoder)")
    p.add_argument("--policy-term", choices=["sum", "final", "final_random"], default="sum",
                   help="divergence utility: sum over the rollout's steps, or its final term only with q = q_max "
                        "(final) or with a random q (final_random)")
    p.add_argument("--tau-max", type=float, default=1.0, help="divergence utility: tau ~ U[0, tau_max]")
    p.add_argument("--tau-anneal-frac", type=float, default=0.0,
                   help="raise tau_max linearly to 1 over this last fraction of the steps")
    p.add_argument("--ckpt-every", type=int, default=10, help="also save the policy to --out every this many epochs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = parse_args(p)
    if args.divergence is None:
        args.divergence = "exact" if args.utility == "divergence" else "hutchinson"
    if args.own_encoder and args.detach_policy_input:
        p.error("--detach-policy-input would leave the policy's own encoder untrained")
    cfg = {**vars(args), "flow_ckpt": str(args.flow_ckpt), "out": str(args.out), "log_csv": str(args.log_csv)}

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    sim = LocationFinding()

    # frozen flow: prior, guidance and history encoder
    ck = torch.load(args.flow_ckpt, map_location=dev)
    if args.own_encoder:
        cfg["policy_encoder"] = {k: ck["config"][k] for k in ENCODER_KEYS if k in ck["config"]}
        cfg["policy_encoder"]["activation"] = args.activation
        if args.policy_enc_hidden_dims is not None:
            cfg["policy_encoder"]["enc_hidden_dims"] = args.policy_enc_hidden_dims
    prior, guidance = build_flow(ck["config"], sim)
    prior.load_state_dict(ck["prior"])
    guidance.load_state_dict(ck["guidance"])
    guidance.to(dev).eval().requires_grad_(False)
    sampler = PosteriorSampler(
        prior, guidance, solver=args.solver, n_steps=args.n_steps, atol=args.tol, rtol=args.tol, divergence=args.divergence
    )
    nfe = [0]  # number of vector field evaluations of the density ODE, for monitoring
    sampler.cnf.register_forward_hook(lambda *_: nfe.__setitem__(0, nfe[0] + 1))

    encoder = build_encoder(cfg["policy_encoder"], sim) if args.own_encoder else guidance.encoder
    policy = Policy(encoder, sim.eps_dim, args.policy_hidden_dims, args.activation, args.design_bound,
                    args.bound_type).to(dev)
    params = list((policy if args.own_encoder else policy.net).parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, div_factor=20.0, final_div_factor=20.0, total_steps=total, pct_start=0.02
    )
    warmup = int(0.002 * total)
    anneal_start = int((1 - args.tau_anneal_frac) * total)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_file = writer = None
    if args.log_csv is not None:
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log_csv, "w", newline="")  # noqa: SIM115 (closed after training)
        writer = csv.writer(log_file)
        writer.writerow(["step", "epoch", "q", "loss", "loss_per_q", "lr", "nfe", "seconds"])

    def save(epoch: int) -> None:
        ck_out = {"config": cfg, "policy": policy.net.state_dict(), "epoch": epoch}
        if args.own_encoder:
            ck_out["encoder"] = policy.encoder.state_dict()
        torch.save(ck_out, args.out)

    t0 = time.time()
    for step in range(total):
        if args.tau_anneal_frac > 0 and step == anneal_start:
            save(step // args.steps_per_epoch)
            shutil.copy(args.out, args.out.with_name(args.out.stem + "_before_anneal.pt"))
        frac = max(0.0, (step - anneal_start) / max(total - anneal_start, 1)) if args.tau_anneal_frac > 0 else 0.0
        tau_max = args.tau_max + (1 - args.tau_max) * frac
        q_max = q_max_at(step, args.q_min, args.Q, warmup)
        q = q_max if args.policy_term == "final" else int(torch.randint(args.q_min, q_max + 1, ()))
        theta = sim.sample_prior(args.batch, dev)
        if args.utility == "divergence":
            # one tau per batch and one theta_0 per sample, shared by all steps of the rollout
            tau = (tau_max * torch.rand(1, device=dev)).expand(args.batch)
            lam = prior.lambda_t(tau)[:, None]
            x = (tau[:, None] * theta + lam * torch.randn_like(theta)).requires_grad_(True)
            probe = torch.randn_like(theta)  # Hutchinson probe, shared by all steps
        # phi of every observation is computed once and stored (encoder.tokens), and the stored tokens are pooled at
        # every step: tok_f for the flow's encoder, tok_o for the policy's own encoder (--own-encoder)
        enc_f, enc_o = guidance.encoder, (policy.encoder if args.own_encoder else None)
        tok_f, tok_o = [], []
        h_f = enc_f.pool(theta.new_zeros(args.batch, 0, enc_f.embed_dim))  # empty history
        h_o = enc_o.pool(theta.new_zeros(args.batch, 0, enc_o.embed_dim)) if enc_o is not None else None
        loss = 0.0
        for t in range(q):
            h_in = h_o if enc_o is not None else (h_f.detach() if args.detach_policy_input else h_f)
            xi = policy.design(h_in)[:, None]  # next design (B, 1, dx), from the history so far
            y_new = sim.simulate(theta, xi)  # reparametrised: gradients reach the design
            tok_f.append(enc_f.tokens(y_new, xi))
            h_f = enc_f.pool(torch.cat(tok_f, dim=1))
            if enc_o is not None:
                tok_o.append(enc_o.tokens(y_new, xi))
                h_o = enc_o.pool(torch.cat(tok_o, dim=1))
            if args.utility == "divergence" and (args.policy_term == "sum" or t == q - 1):
                w = guidance(x, tau, h_f)
                if args.divergence == "exact":
                    div = exact_divergence(w, x, create_graph=True)
                else:
                    div = hutchinson_divergence(w, x, probe, create_graph=True)
                loss = loss + div.mean()
        if args.utility == "density":
            nfe[0] = 0
            loss = -sampler.log_prob(theta, h_f).mean()

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
                save(epoch)
            extra = f"  nfe {nfe[0]}" if args.utility == "density" else ""
            mem = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0
            print(f"[policy-{args.utility}] epoch {epoch:4d}  q {q:2d}  loss {loss.item():.4f}{extra}  "
                  f"peak mem {mem:.2f} GB  ({time.time() - t0:.0f}s)", flush=True)

    save(args.epochs)
    if log_file is not None:
        log_file.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
