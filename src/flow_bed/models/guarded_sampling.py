"""Posterior sampling and log-density evaluation with guards against adaptive-solver hangs.

torchdyn's adaptive loop (dopri5) has no step limit and no minimum step size: if one sample of a batch makes the error
estimate non-finite (NaN dt) or keeps it huge until dt underflows to 0, the ``while t < T`` loop never ends, and the
whole batch hangs with it. This happens for points far outside the posterior (q ~ 0), where the backward ODE blows up.
Three guards, from coarse to fine:

``run_capped`` / ``guarded_sample``
    Forward sampling with a call cap: a chunk that stalls (non-finite vector field or more than ``cap`` calls) is
    redrawn (the stall rate is ~1 per 20k chunks or less, so the redraw changes the sample negligibly).
``run_capped_log_prob`` / ``guarded_log_prob``
    Backward log q with a call cap: a chunk that stalls is bisected down to the single points that stall on their own,
    which get log q = -inf. Correct but slow when many points stall (each isolation costs several capped solves).
``frozen_log_prob``
    Backward log q with a freeze guard: a point whose solve heads off (non-finite state, velocity or divergence, or
    state outside the ball |x| <= R) is frozen for the rest of the solve -- its velocity and divergence are set to 0,
    so it no longer drives the batch's adaptive step size -- and gets log q = -inf (density 0) at the end. One pass,
    no reruns; the call cap stays only as a backstop (raises ``SolverStall``). The rule is fixed per point, so for
    InfoNCE it only defines a (slightly different) critic and the bound stays valid; a frozen true pair can only
    lower it.
"""

from __future__ import annotations

import torch
from torch import Tensor

from flow_bed.models.posterior_sampler import PosteriorSampler


class SolverStall(Exception):
    def __init__(self, reason: str, n_calls: int, ts: list[float], x: Tensor, out: Tensor | None = None):
        super().__init__(f"{reason} after {n_calls} vector-field calls, t {ts[-1] if ts else float('nan')}")
        self.reason, self.n_calls, self.ts, self.x, self.out = reason, n_calls, ts, x, out


@torch.no_grad()
def run_capped(sampler: PosteriorSampler, x0: Tensor, history: Tensor, cap: int = 2000) -> tuple[Tensor, int]:
    """Forward sampling ODE (t: 0 -> 1) from x0 (n, d) with one history row per sample (n, out_dim) -> (x1, calls).
    Raises SolverStall on a non-finite vector-field value or after ``cap`` calls."""
    vf = sampler.vf
    calls = {"n": 0, "t": []}
    orig = vf.forward

    def counted(t, x, args=None):
        calls["n"] += 1
        calls["t"].append(float(t))
        if calls["n"] > cap:
            raise SolverStall("cap", calls["n"], calls["t"], x.detach())
        out = orig(t, x)
        if not torch.isfinite(out).all():
            raise SolverStall("nonfinite", calls["n"], calls["t"], x.detach(), out.detach())
        return out

    vf.forward = counted
    try:
        sampler._set_history(history)
        _, sol = sampler.node(x0, torch.tensor([0.0, 1.0], device=x0.device, dtype=x0.dtype))
        return sol[-1], calls["n"]
    finally:
        del vf.forward  # back to the class method


def guarded_sample(sampler: PosteriorSampler, n: int, history: Tensor, chunk: int = 500, cap: int = 2000,
                   log=print, max_redraws: int = 50) -> tuple[Tensor, int]:
    """n samples for one encoded history (1, out_dim) -> (theta (n, d), number of redrawn chunks)."""
    ref = next(sampler.vf.prior.parameters())
    out, done, redraws = [], 0, 0
    while done < n:
        b = min(chunk, n - done)
        x0 = torch.randn(b, sampler.dim, device=ref.device, dtype=ref.dtype)
        try:
            x1, _ = run_capped(sampler, x0, history.expand(b, -1), cap)
        except SolverStall as e:
            redraws += 1
            log(f"guard: chunk at {done}/{n} stalled ({e}), redrawn", flush=True)
            if redraws > max_redraws:
                raise RuntimeError(f"more than {max_redraws} stalled chunks") from e
            continue
        out.append(x1)
        done += b
    return torch.cat(out), redraws


@torch.no_grad()
def run_capped_log_prob(sampler: PosteriorSampler, theta: Tensor, history: Tensor, cap: int = 1000) -> Tensor:
    """log q(theta | history) by the backward ODE (sampler.log_prob), stopped with SolverStall after ``cap``
    vector-field calls (dopri5 can crawl or loop forever on points far out, where q ~ 0)."""
    cnf = sampler.cnf
    calls = {"n": 0, "t": []}
    orig = cnf.forward

    def counted(t, x, args=None):
        calls["n"] += 1
        calls["t"].append(float(t))
        if calls["n"] > cap:
            raise SolverStall("cap", calls["n"], calls["t"], x.detach())
        return orig(t, x)

    cnf.forward = counted
    try:
        return sampler.log_prob(theta, history)
    finally:
        del cnf.forward


def guarded_log_prob(sampler: PosteriorSampler, theta: Tensor, history: Tensor, chunk: int = 10_000,
                     cap: int = 1000) -> tuple[Tensor, int]:
    """log q(theta | history) for theta (n, d) and one encoded history (1, out_dim), in chunks; a chunk that stalls
    is bisected down to the single points that stall on their own, which get log q = -inf (density 0, as for the
    points where the solve ends non-finite). Returns (log q (n,), number of stalled points)."""
    out = torch.empty(theta.shape[0], device=theta.device, dtype=theta.dtype)
    n_stalled = 0

    def solve(ids: Tensor) -> None:
        nonlocal n_stalled
        try:
            out[ids] = run_capped_log_prob(sampler, theta[ids], history.expand(len(ids), -1), cap)
        except SolverStall:
            if len(ids) == 1:
                out[ids] = -float("inf")
                n_stalled += 1
                return
            solve(ids[: len(ids) // 2])
            solve(ids[len(ids) // 2:])

    for ids in torch.arange(theta.shape[0], device=theta.device).split(chunk):
        solve(ids)
    return out, n_stalled


@torch.no_grad()
def frozen_log_prob(sampler: PosteriorSampler, theta: Tensor, history: Tensor | None, R: float = 20.0,
                    cap: int = 3000) -> tuple[Tensor, Tensor, int]:
    """log q(theta | history) with the freeze guard, for theta (B, d) and one history row per point (B, out_dim),
    or ``history=None`` for the prior flow. Returns (log q (B,), frozen mask (B,), vector-field calls)."""
    cnf = sampler.cnf
    orig = cnf.forward
    frozen = torch.zeros(theta.shape[0], dtype=torch.bool, device=theta.device)
    calls = {"n": 0}

    def guarded(t, x, args=None):
        calls["n"] += 1
        if calls["n"] > cap:
            raise SolverStall("cap", calls["n"], [float(t)], x.detach())
        out = orig(t, x)
        bad = ~torch.isfinite(out).all(-1) | ~torch.isfinite(x).all(-1) | (x[:, :-1].norm(dim=-1) > R)
        frozen.logical_or_(bad)
        return torch.where(frozen[:, None], torch.zeros_like(out), out)

    cnf.forward = guarded
    try:
        lq = sampler.log_prob(theta, history)
    finally:
        del cnf.forward
    lq = torch.where(frozen | ~torch.isfinite(lq), torch.full_like(lq, -float("inf")), lq)
    return lq, frozen, calls["n"]
