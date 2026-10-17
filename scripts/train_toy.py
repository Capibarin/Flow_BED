"""Train the prior flow and the likelihood-guidance model on the toy linear-Gaussian problem.

    python scripts/train_toy.py --steps 3000 --out checkpoints/toy.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flow_bed.models import LikelihoodGuidanceModel, PriorFlowModel
from flow_bed.models.history_encoder import HistoryEncoder
from flow_bed.simulators import LinearGaussian


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--max-obs", type=int, default=4, help="train with 1..max-obs observations per batch")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128, 128])
    p.add_argument("--out", type=Path, default=Path("checkpoints/toy.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    sim = LinearGaussian(args.dim, args.noise)

    # 1) prior flow: velocity + score
    prior = PriorFlowModel(args.dim, hidden_dims=args.hidden_dims).to(dev)
    opt = torch.optim.Adam(prior.parameters(), lr=args.lr)
    for step in range(args.steps):
        loss = prior.shared_step(sim.sample_prior(args.batch, dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"[prior]    step {step:5d}  loss {loss.item():.4f}")

    # 2) guidance on top of the frozen prior, random number of observations per batch
    encoder = HistoryEncoder(y_dim=1, eps_dim=args.dim, hidden_dims=args.hidden_dims)
    guidance = LikelihoodGuidanceModel(prior, encoder, hidden_dims=args.hidden_dims).to(dev)
    opt = torch.optim.Adam([p for p in guidance.parameters() if p.requires_grad], lr=args.lr)  # encoder + net
    for step in range(args.steps):
        n_obs = int(torch.randint(1, args.max_obs + 1, ()))
        loss = guidance.shared_step(sim.sample_joint(args.batch, n_obs, dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"[guidance] step {step:5d}  loss {loss.item():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"args": {**vars(args), "out": str(args.out)}, "prior": prior.state_dict(), "guidance": guidance.state_dict()},
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
