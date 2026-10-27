"""Stage 1 for location finding: train the prior flow, then the history encoder + likelihood guidance on random designs.

Mirrors JADAI's pre-training phase (K = 2 sources in 2D, theta ~ N(0, I), random designs eps ~ N(0, I)):
50 epochs x 781 steps (200k samples / batch 256), Adam with a one-cycle schedule (max lr 1e-3), gradient clipping
at 5, and the number of observations per batch q ~ U{q_min, ..., q_max}, with q_max ramped from q_min to Q.
--prior-epochs trains the prior for a different number of epochs than the guidance; --normalize-y standardises the
observations in the history encoder with their mean and std under random designs (stored in the config).
The policy is trained afterwards with scripts/train_lf_policy.py.

    python scripts/train_lf_flow.py --Q 30 --out checkpoints/lf_flow.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from flow_bed.models import LikelihoodGuidanceModel, PriorFlowModel
from flow_bed.models.flow_matching import LEGACY_ACTIVATION
from flow_bed.models.history_encoder import HistoryEncoder, SelfAttentionHistoryEncoder
from flow_bed.simulators.location_finding import LocationFinding


def build_encoder(cfg: dict, sim: LocationFinding) -> HistoryEncoder | SelfAttentionHistoryEncoder:
    """History encoder from a config dict (the parsed arguments)."""
    y_norm = {"y_loc": cfg.get("y_loc", 0.0), "y_scale": cfg.get("y_scale", 1.0)}
    act = cfg.get("activation", LEGACY_ACTIVATION)
    if cfg["encoder"] == "deepsets":
        return HistoryEncoder(sim.y_dim, sim.eps_dim, cfg["enc_hidden_dims"], cfg["embed_dim"], **y_norm, activation=act)
    return SelfAttentionHistoryEncoder(
        sim.y_dim, sim.eps_dim, cfg["enc_hidden_dims"], cfg["embed_dim"], cfg["attn_layers"], cfg["attn_heads"],
        **y_norm, activation=act, ff_dim=cfg.get("attn_ff_dim"),
    )


def build_guidance(cfg: dict, sim: LocationFinding, prior: PriorFlowModel) -> LikelihoodGuidanceModel:
    """History encoder + guidance from a config dict (the parsed arguments). Freezes ``prior``."""
    hidden_dims = cfg.get("guidance_hidden_dims") or cfg["hidden_dims"]
    return LikelihoodGuidanceModel(
        prior, build_encoder(cfg, sim), hidden_dims=hidden_dims, parametrization=cfg.get("param", "w"),
        activation=cfg.get("activation", LEGACY_ACTIVATION),
    )


def build_flow(cfg: dict, sim: LocationFinding) -> tuple[PriorFlowModel, LikelihoodGuidanceModel]:
    """Prior flow and guidance, e.g. to load a checkpoint of this script."""
    prior = PriorFlowModel(sim.dim, hidden_dims=cfg["hidden_dims"], activation=cfg.get("activation", LEGACY_ACTIVATION))
    return prior, build_guidance(cfg, sim, prior)


def q_max_at(step: int, q_min: int, Q: int, warmup: int) -> int:
    """JADAI's QScheduler (cosine): q_max goes from q_min to Q over ``warmup`` steps."""
    frac = min(1.0, step / max(warmup, 1))
    return q_min + int((1 - math.cos(frac * math.pi)) / 2 * (Q - q_min))


def train(name, params, loss_fn, total_steps, lr, steps_per_epoch):
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
        if (step + 1) % steps_per_epoch == 0:
            epoch = (step + 1) // steps_per_epoch
            print(f"[{name}] epoch {epoch:4d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--Q", type=int, default=30, help="maximum number of observations (time horizon)")
    p.add_argument("--q-min", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--prior-epochs", type=int, default=None, help="epochs for the prior (default: --epochs)")
    p.add_argument("--normalize-y", action="store_true", help="standardise y in the history encoder")
    p.add_argument("--param", choices=["w", "h"], default="w", help="guidance output parametrisation")
    p.add_argument("--steps-per-epoch", type=int, default=781, help="200k samples / batch 256 as in JADAI")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--guidance-hidden-dims", type=int, nargs="+", default=None,
                   help="hidden layers of the guidance network (default: --hidden-dims)")
    p.add_argument("--encoder", choices=["deepsets", "attention"], default="deepsets")
    p.add_argument("--enc-hidden-dims", type=int, nargs="+", default=[128, 128])
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--attn-layers", type=int, default=4)
    p.add_argument("--attn-heads", type=int, default=2)
    p.add_argument("--activation", choices=["silu", "selu"], default="silu", help="activation of all MLPs")
    p.add_argument("--out", type=Path, default=Path("checkpoints/lf_flow.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = {**vars(args), "out": str(args.out)}

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    sim = LocationFinding()
    if args.normalize_y:
        _, _, y = sim.sample_joint(1_000_000, 1, dev)  # observations under random designs
        cfg["y_loc"], cfg["y_scale"] = y.mean().item(), y.std().item()
        print(f"y standardisation: loc {cfg['y_loc']:.3f}  scale {cfg['y_scale']:.3f}", flush=True)
    prior = PriorFlowModel(sim.dim, hidden_dims=args.hidden_dims, activation=args.activation).to(dev)
    total = args.epochs * args.steps_per_epoch
    warmup = int(0.002 * total)
    prior_total = (args.prior_epochs or args.epochs) * args.steps_per_epoch

    # 1) prior flow on fresh prior samples
    train("prior", list(prior.parameters()), lambda step: prior.shared_step(sim.sample_prior(args.batch, dev)),
          prior_total, args.lr, args.steps_per_epoch)
    prior.eval()

    # 2) encoder + guidance on histories of q random designs (the prior is frozen inside the guidance model)
    guidance = build_guidance(cfg, sim, prior).to(dev)
    def guidance_loss(step):
        q = int(torch.randint(args.q_min, q_max_at(step, args.q_min, args.Q, warmup) + 1, ()))
        return guidance.shared_step(sim.sample_joint(args.batch, q, dev))

    train("guidance", [p for p in guidance.parameters() if p.requires_grad], guidance_loss,
          total, args.lr, args.steps_per_epoch)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "prior": prior.state_dict(), "guidance": guidance.state_dict()}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
