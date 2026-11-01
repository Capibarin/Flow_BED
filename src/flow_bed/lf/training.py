"""Training loop and observation-count schedule shared by the location-finding training scripts."""

from __future__ import annotations

import math
import time

import torch


def q_max_at(step: int, q_min: int, Q: int, warmup: int) -> int:
    """Cosine schedule: q_max goes from q_min to Q over ``warmup`` steps."""
    frac = min(1.0, step / max(warmup, 1))
    return q_min + int((1 - math.cos(frac * math.pi)) / 2 * (Q - q_min))


def train(name, params, loss_fn, total_steps, lr, steps_per_epoch, on_epoch=None):
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
            if on_epoch is not None:
                on_epoch(epoch)
