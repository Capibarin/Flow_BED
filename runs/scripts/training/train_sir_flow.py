"""SIR stage 1: prior flow on z (filtered prior, from the path bank), then history encoder + guidance on histories of
q ~ U{q_min, ..., q_max} independent uniform measurement times (q_max ramped to Q as in train_lf_flow.py).
Same architecture and schedule as runs/configs/lf/flow_attn.yaml. The bank (--bank paths) is re-simulated every
--refresh steps.
Options for the posterior variants (defaults = the original flow):
  --train-designs mix   training designs per sample: 1/2 independent uniform times (as before), 1/4 increasing random
                        designs (iDAD's increasing transform with u ~ U(-5, 5), as iDAD's random baseline), 1/4 equally
                        spaced designs with a random start U(LB, 50) and spacing U(1, (UB - start) / (Q - 1));
                        generic, not tied to any trained policy;
  --ordered-encoder     SirAttentionEncoder with sorted pairs, differences to the predecessor, rank embedding;
  --y-transform sqrt    sqrt(y + 1) before the standardisation;
  --attn-activation silu  SiLU instead of ReLU in the attention layers' feed-forward part;
  --t-sampling mix      flow time t of the guidance loss from 1/2 U(0, 1) + 1/2 Beta(4, 1) (more weight near t = 1, the
                        posterior end), clamped to <= 1 - 1e-4 (the target is 0/0 at t = 1);
  --ema-decay D         also keep an exponential moving average of the guidance weights (saved as "guidance_ema"
                        next to the raw "guidance"; load with load_flow(..., ema=True) or PATH@ema);
  --prior-from CKPT     reuse the prior flow of CKPT (no prior training: the variants share the same prior);
  --max-steps N         stop the guidance training after N steps (smoke tests).
--config file.yaml sets the defaults of these arguments (see runs/configs/sir/flow.yaml and flow_bed.config).
    python runs/scripts/training/train_sir_flow.py --config runs/configs/sir/flow.yaml --out CKPT.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from flow_bed.config import parse_args
from flow_bed.lf.training import q_max_at, train
from flow_bed.models import PriorFlowModel
from flow_bed.simulators import sir
from flow_bed.sir import SPEC, build_sir_guidance


def train_ema(name, params, loss_fn, total_steps, lr, steps_per_epoch, on_epoch, ema, decay):
    """flow_bed.lf.training.train (same optimiser, one-cycle schedule, clipping) + an EMA of ``params`` into ``ema``
    after every optimiser step (if ``ema`` is not None)."""
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, div_factor=1e5, final_div_factor=2.0, total_steps=total_steps, pct_start=0.08
    )
    t0 = time.time()
    for step in range(total_steps):
        loss = loss_fn(step)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                torch._foreach_mul_(ema, decay)
                torch._foreach_add_(ema, [q.detach() for q in params], alpha=1.0 - decay)
        if (step + 1) % steps_per_epoch == 0:
            epoch = (step + 1) // steps_per_epoch
            print(f"[{name}] epoch {epoch:4d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
            if on_epoch is not None:
                on_epoch(epoch)


def guidance_step_tmix(guidance, batch):
    """LikelihoodGuidanceModel.shared_step with t ~ 1/2 U(0, 1) + 1/2 Beta(4, 1) (= U^(1/4)), t <= 1 - 1e-4."""
    theta1, eps, y = batch
    B, dev = theta1.shape[0], theta1.device
    t = torch.where(torch.rand(B, 1, device=dev) < 0.5, torch.rand(B, 1, device=dev),
                    torch.rand(B, 1, device=dev) ** 0.25).clamp(max=1 - 1e-4)
    theta0 = torch.randn_like(theta1)
    lam, lam_sp = guidance.prior.lambda_t(t), guidance.prior.lambda_sp_t(t)
    theta_t = t * theta1 + lam * theta0
    u_t = theta1 + lam_sp / lam * theta0
    c_t = lam**2 - lam_sp * t
    with torch.no_grad():
        _, s_prior = guidance.prior(theta_t, t[:, 0])
    w = guidance(theta_t, t[:, 0], guidance.encoder(y, eps))
    return F.mse_loss(c_t * s_prior + t * w, t * u_t - theta_t)


def mixed_taus(b: int, q: int, Q: int, gen: torch.Generator, device) -> torch.Tensor:
    """--train-designs mix: (b, q) designs, the first q of Q; per sample 1/2 independent uniform, 1/4 increasing random
    (iDAD's transform, u ~ U(-5, 5)), 1/4 equally spaced with a random start and spacing."""
    kind = torch.randint(0, 4, (b, 1), device=device, generator=gen)
    uni = sir.uniform_taus(b, q, gen, device)
    inc, prev = [], torch.full((b,), sir.LB, device=device)
    for _ in range(q):
        prev = sir.next_tau(torch.rand(b, device=device, generator=gen) * 10 - 5, prev)
        inc.append(prev)
    inc = torch.stack(inc, 1)
    start = sir.LB + torch.rand(b, 1, device=device, generator=gen) * (50.0 - sir.LB)
    max_sp = (sir.UB - start) / (Q - 1)
    sp = 1.0 + torch.rand(b, 1, device=device, generator=gen) * (max_sp - 1.0)
    grid = start + sp * torch.arange(q, device=device)[None]
    return torch.where(kind < 2, uni, torch.where(kind == 2, inc, grid))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--Q", type=int, default=5)
    p.add_argument("--q-min", type=int, default=1)
    p.add_argument("--prior-epochs", type=int, default=50)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--guidance-hidden-dims", type=int, nargs="+", default=[512, 512, 512, 512])
    p.add_argument("--enc-hidden-dims", type=int, nargs="+", default=[256, 256])
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--attn-layers", type=int, default=4)
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--attn-ff-dim", type=int, default=256)
    p.add_argument("--bank", type=int, default=100_000)
    p.add_argument("--refresh", type=int, default=10_000)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--train-designs", choices=["uniform", "mix"], default="uniform")
    p.add_argument("--ordered-encoder", action="store_true")
    p.add_argument("--y-transform", choices=["linear", "sqrt"], default="linear")
    p.add_argument("--prior-from", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--attn-activation", choices=["relu", "silu"], default="relu")
    p.add_argument("--t-sampling", choices=["uniform", "mix"], default="uniform")
    p.add_argument("--ema-decay", type=float, default=0.0)
    args = parse_args(p)
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    cfg = {**vars(args), "out": str(args.out), "prior_from": str(args.prior_from), "encoder": "attention", "param": "h",
           "activation": "silu", "model": "sir"}
    prior_ck = None
    if args.prior_from is not None:
        prior_ck = torch.load(args.prior_from, map_location=dev)
        cfg["hidden_dims"] = prior_ck["config"]["hidden_dims"]

    bank = sir.PathBank(args.bank, args.seed, dev)
    g = bank.gen
    y = bank.observe(bank.sample_idx(100_000), sir.uniform_taus(100_000, 1, g, dev))
    if args.y_transform == "sqrt":
        y = (y.clamp(min=0.0) + 1.0).sqrt()  # standardisation of the transformed y (SirAttentionEncoder)
    cfg["y_loc"], cfg["y_scale"] = y.mean().item(), y.std().item()
    print(f"y standardisation: loc {cfg['y_loc']:.2f}  scale {cfg['y_scale']:.2f}  (kept {bank.valid.shape[0]} paths)",
          flush=True)

    def maybe_refresh(step):
        if step > 0 and step % args.refresh == 0:
            bank.refresh()

    prior = PriorFlowModel(SPEC.dim, hidden_dims=cfg["hidden_dims"], activation="silu").to(dev)

    def prior_loss(step):
        maybe_refresh(step)
        return prior.shared_step(bank.z[bank.sample_idx(args.batch)])

    if prior_ck is not None:
        prior.load_state_dict(prior_ck["prior"])
        print(f"prior loaded from {args.prior_from}", flush=True)
    else:
        train("prior", list(prior.parameters()), prior_loss, args.prior_epochs * args.steps_per_epoch, args.lr,
              args.steps_per_epoch)
    prior.eval()

    guidance = build_sir_guidance(cfg, prior).to(dev)
    total = args.epochs * args.steps_per_epoch if args.max_steps is None else args.max_steps
    warmup = int(0.002 * total)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    trainable = [(n, q) for n, q in guidance.named_parameters() if q.requires_grad]
    ema = [q.detach().clone() for _, q in trainable] if args.ema_decay > 0 else None

    def save(epoch):
        ck = {"config": cfg, "prior": prior.state_dict(), "guidance": guidance.state_dict(), "epoch": epoch}
        if ema is not None:
            sd = dict(ck["guidance"])
            sd.update({n: e.clone() for (n, _), e in zip(trainable, ema)})
            ck["guidance_ema"] = sd
        torch.save(ck, args.out)

    def guidance_loss(step):
        maybe_refresh(step)
        q = int(torch.randint(args.q_min, q_max_at(step, args.q_min, args.Q, warmup) + 1, ()))
        idx = bank.sample_idx(args.batch)
        tau = (sir.uniform_taus(args.batch, q, g, dev) if args.train_designs == "uniform"
               else mixed_taus(args.batch, q, args.Q, g, dev))
        batch = (bank.z[idx], sir.tau_to_eps(tau)[..., None], bank.observe(idx, tau))
        return guidance.shared_step(batch) if args.t_sampling == "uniform" else guidance_step_tmix(guidance, batch)

    def on_epoch(epoch):
        if epoch % args.ckpt_every == 0 and epoch < args.epochs:
            save(epoch)

    train_ema("guidance", [q for _, q in trainable], guidance_loss, total, args.lr, args.steps_per_epoch, on_epoch,
              ema, args.ema_decay)
    save(args.epochs)
    print(f"saved {args.out}  (peak GPU memory {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GB allocated, "
          f"{torch.cuda.max_memory_reserved(dev) / 2**30:.2f} GB reserved)")


if __name__ == "__main__":
    main()
