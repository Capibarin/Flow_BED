"""SIR builders: load a trained SIR flow checkpoint and roll out a policy's designs on a path bank."""

from __future__ import annotations

import torch

from flow_bed.simulators import sir
from flow_bed.sir.encoders import SPEC, build_sir_flow

__all__ = ["SPEC", "load_flow", "rollout_designs"]


def load_flow(path, dev, ema: bool = False):
    """Flow checkpoint -> (prior, guidance, config); ``ema``: the EMA weights of the guidance (train_sir_flow.py
    --ema-decay), else the raw weights. A path ending in "@ema" also selects the EMA weights."""
    path = str(path)
    if path.endswith("@ema"):
        path, ema = path[:-4], True
    ck = torch.load(path, map_location=dev)
    prior, guidance = build_sir_flow(ck["config"])  # original modules for configs without the SIR options
    prior.load_state_dict(ck["prior"])
    guidance.load_state_dict(ck["guidance_ema"] if ema else ck["guidance"])
    return prior.to(dev).eval(), guidance.to(dev).eval().requires_grad_(False), ck["config"]


def rollout_designs(policy, enc, bank, idx, T, detach_input=False, w=1):
    """Policy rollout of T steps on the paths idx: (tau (B, T), y (B, T, 1), list of encoded histories h_1..h_T)."""
    B = idx.shape[0]
    tau_prev = torch.full((B, 1), sir.LB, device=idx.device)
    toks, hs, taus, ys = [], [], [], []
    h = enc.pool(tau_prev.new_zeros(B, 0, enc.embed_dim))
    for _ in range(T):
        tau = sir.next_tau(policy.design(h.detach() if detach_input else h), tau_prev)
        y = bank.observe(idx, tau, w)
        toks.append(enc.tokens(y, sir.tau_to_eps(tau)[..., None]))
        h = enc.pool(torch.cat(toks, dim=1))
        hs.append(h)
        taus.append(tau)
        ys.append(y)
        tau_prev = tau
    return torch.cat(taus, 1), torch.cat(ys, 1), hs
