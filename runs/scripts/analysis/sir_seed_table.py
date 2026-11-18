"""Seed table from eval_sir_infonce.py per-run files: mean InfoNCE per setting with the seed spread, the paired
difference to a reference design set, and optional paired differences between settings.

The per-run .pt files hold {name: {"infonce": (runs,), ...}} on the same held-out runs in the same order, so values
pair across files. Names ending in _s<seed> are grouped into one setting (seed mean); other names are single rows.
The reference (--ref, default idad: the seed mean of idad_s*) gives the "vs ref" column: the per-run difference of
the setting's seed-mean to the reference's, with its standard error.
    sir_seed_table.py eval_a.pt eval_b.pt ... [--ref idad] [--pairs A,B ...]
"""

from __future__ import annotations

import argparse
import math
import re

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="per-run .pt files of eval_sir_infonce.py")
    p.add_argument("--ref", default="idad", help="reference setting (seed group or single name)")
    p.add_argument("--pairs", nargs="*", default=[], help="A,B: also print the paired difference A - B")
    args = p.parse_args()

    values: dict[str, torch.Tensor] = {}
    for f in args.files:
        for name, v in torch.load(f, map_location="cpu", weights_only=False).items():
            values[name] = v["infonce"].double()

    groups: dict[str, list[str]] = {}
    for name in values:
        groups.setdefault(re.sub(r"_s\d+$", "", name), []).append(name)
    mean_of = {g: torch.stack([values[k] for k in ks]).mean(0) for g, ks in groups.items()}
    if args.ref not in mean_of:
        p.error(f"reference '{args.ref}' not among {sorted(groups)}")
    ref = mean_of[args.ref]

    print(f"{'setting':44s} n  mean    seed-sd  vs {args.ref} (paired +- s.e.)   per seed")
    for g in sorted(groups, key=lambda g: -mean_of[g].mean().item()):
        ks = sorted(groups[g])
        seeds = torch.tensor([values[k].mean().item() for k in ks])
        d = mean_of[g] - ref
        sd = f"{seeds.std().item():.3f}" if len(ks) > 1 else "  -  "
        per_seed = " ".join(f"{s:.3f}" for s in seeds)
        print(f"{g:44s} {len(ks)}  {seeds.mean():.3f}  {sd}   {d.mean():+.3f} +- "
              f"{d.std().item() / math.sqrt(d.numel()):.3f}   [{per_seed}]")

    if args.pairs:
        print("\npaired differences (seed-averaged):")
        for pair in args.pairs:
            a, b = pair.split(",")
            d = mean_of[a] - mean_of[b]
            print(f"{a} - {b}: {d.mean():+.3f} +- {d.std().item() / math.sqrt(d.numel()):.3f}")


if __name__ == "__main__":
    main()
