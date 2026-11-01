"""Stage 1 for location finding: train the prior flow, then the history encoder + likelihood guidance on random designs.

Pre-training on random designs (K = 2 sources in 2D, theta ~ N(0, I), random designs eps ~ N(0, I)):
50 epochs x 781 steps (200k samples / batch 256), Adam with a one-cycle schedule (max lr 1e-3), gradient clipping
at 5, and the number of observations per batch q ~ U{q_min, ..., q_max}, with q_max ramped from q_min to Q.
--prior-epochs trains the prior for a different number of epochs than the guidance; --normalize-y standardises the
observations in the history encoder with their mean and std under random designs (stored in the config).
--init-prior skips the prior's training and uses the (frozen) prior of a checkpoint of this script instead (or of any
checkpoint with "config" and "prior", e.g. a prior-only one); its architecture comes from that checkpoint, and the
random seed is reset before the guidance is built, so the guidance does not depend on how the prior was obtained.
--prior-only trains the prior alone and saves it (config + prior) to --out, for a later --init-prior.
--sigma and --m set the outcome model y ~ N(log(b + sum_k a / (m + |xi - theta_k|^2)), sigma^2) (stored in the config;
the policy scripts take it from the flow's checkpoint, see build_sim).
--ckpt-every n also saves the model to --out every n epochs of the guidance's training.
--config file.yaml sets the defaults of these arguments (see runs/configs/lf and flow_bed.config).
The policy is trained afterwards with runs/scripts/training/train_lf_policy.py or runs/scripts/training/train_lf_joint.py.

    python runs/scripts/training/train_lf_flow.py --config runs/configs/lf/flow_attn.yaml --out checkpoints/lf_flow_attn.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flow_bed.config import parse_args
from flow_bed.lf.builders import build_guidance, build_sim
from flow_bed.lf.training import q_max_at, train
from flow_bed.models import PriorFlowModel
from flow_bed.models.flow_matching import LEGACY_ACTIVATION


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--Q", type=int, default=30, help="maximum number of observations (time horizon)")
    p.add_argument("--q-min", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--prior-epochs", type=int, default=None, help="epochs for the prior (default: --epochs)")
    p.add_argument("--normalize-y", action="store_true", help="standardise y in the history encoder")
    p.add_argument("--param", choices=["w", "h"], default="w", help="guidance output parametrisation")
    p.add_argument("--steps-per-epoch", type=int, default=781, help="200k samples / batch 256")
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
    p.add_argument("--attn-ff-dim", type=int, default=None,
                   help="feed-forward width of the attention layers (default: 2 x --embed-dim)")
    p.add_argument("--activation", choices=["silu", "selu"], default="silu", help="activation of all MLPs")
    p.add_argument("--sigma", type=float, default=0.5, help="observation noise std of y")
    p.add_argument("--m", type=float, default=1e-4, help="offset m in a / (m + |xi - theta_k|^2)")
    p.add_argument("--init-prior", type=Path, default=None,
                   help="use the frozen prior of this checkpoint instead of training one (see above)")
    p.add_argument("--prior-only", action="store_true", help="train and save the prior alone (see above)")
    p.add_argument("--out", type=Path, default=Path("checkpoints/lf_flow.pt"))
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="also save the model to --out every this many epochs of the guidance's training")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = parse_args(p)
    if args.prior_only and args.init_prior is not None:
        p.error("--prior-only trains a prior; it cannot start from --init-prior")
    cfg = {**vars(args), "out": str(args.out), "init_prior": str(args.init_prior)}

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    sim = build_sim(cfg)
    if args.normalize_y:
        _, _, y = sim.sample_joint(1_000_000, 1, dev)  # observations under random designs
        cfg["y_loc"], cfg["y_scale"] = y.mean().item(), y.std().item()
        print(f"y standardisation: loc {cfg['y_loc']:.3f}  scale {cfg['y_scale']:.3f}", flush=True)
    total = args.epochs * args.steps_per_epoch
    warmup = int(0.002 * total)

    if args.init_prior is not None:
        ck = torch.load(args.init_prior, map_location=dev)
        if ck["config"].get("activation", LEGACY_ACTIVATION) != args.activation:
            p.error(f"--activation {args.activation} differs from the prior's ({ck['config'].get('activation')})")
        cfg["hidden_dims"] = ck["config"]["hidden_dims"]
        prior = PriorFlowModel(sim.dim, hidden_dims=cfg["hidden_dims"], activation=args.activation).to(dev)
        prior.load_state_dict(ck["prior"])
        torch.manual_seed(args.seed)
    else:
        # 1) prior flow on fresh prior samples
        prior = PriorFlowModel(sim.dim, hidden_dims=args.hidden_dims, activation=args.activation).to(dev)
        prior_total = (args.prior_epochs or args.epochs) * args.steps_per_epoch
        train("prior", list(prior.parameters()), lambda step: prior.shared_step(sim.sample_prior(args.batch, dev)),
              prior_total, args.lr, args.steps_per_epoch)
        if args.prior_only:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": cfg, "prior": prior.state_dict()}, args.out)
            print(f"saved {args.out}")
            return
    prior.eval()

    # 2) encoder + guidance on histories of q random designs (the prior is frozen inside the guidance model)
    guidance = build_guidance(cfg, sim, prior).to(dev)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save(epoch):
        torch.save({"config": cfg, "prior": prior.state_dict(), "guidance": guidance.state_dict(), "epoch": epoch},
                   args.out)

    def guidance_loss(step):
        q = int(torch.randint(args.q_min, q_max_at(step, args.q_min, args.Q, warmup) + 1, ()))
        return guidance.shared_step(sim.sample_joint(args.batch, q, dev))

    def on_epoch(epoch):
        if args.ckpt_every and epoch % args.ckpt_every == 0 and epoch < args.epochs:
            save(epoch)

    train("guidance", [p for p in guidance.parameters() if p.requires_grad], guidance_loss,
          total, args.lr, args.steps_per_epoch, on_epoch)
    save(args.epochs)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
