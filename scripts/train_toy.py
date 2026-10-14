"""Train the prior flow and the likelihood-guidance model on the toy linear-Gaussian problem.

    python scripts/train_toy.py --steps 3000 --out checkpoints/toy.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flow_bed.models import LikelihoodGuidanceModel, PriorFlowModel
from flow_bed.simulators import LinearGaussian


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--max-obs", type=int, default=4, help="train with 1..max-obs observations per theta")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--out", type=Path, default=Path("checkpoints/toy.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    sim = LinearGaussian(args.dim, args.noise)

    prior = PriorFlowModel(args.dim, hidden=args.hidden).to(dev)
    guidance = LikelihoodGuidanceModel(args.dim, y_dim=1, eps_dim=args.dim, hidden=args.hidden).to(dev)

    # 1) prior flow --------------------------------------------------------------------
    opt = torch.optim.Adam(prior.parameters(), lr=args.lr)
    for step in range(args.steps):
        theta = sim.sample_prior(args.batch, dev)
        loss = prior.loss(theta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"[prior]    step {step:5d}  loss {loss.item():.4f}")

    # 2) guidance ----------------------------------------------------------------------
    prior.eval().requires_grad_(False)
    opt = torch.optim.Adam(guidance.parameters(), lr=args.lr)
    for step in range(args.steps):
        theta, eps, y = sim.sample_joint(args.batch, args.max_obs, dev)
        # random number of valid observations per sample -> mask
        n_valid = torch.randint(1, args.max_obs + 1, (args.batch, 1), device=dev)
        mask = torch.arange(args.max_obs, device=dev)[None] < n_valid
        y = y * mask[..., None]  # masked entries are irrelevant, just keep them finite
        loss = guidance.loss(prior, theta, y, eps, mask=mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"[guidance] step {step:5d}  loss {loss.item():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"args": vars(args), "prior": prior.state_dict(), "guidance": guidance.state_dict()},
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
