"""SIR-specific history encoder options and the flow builder that knows them.

SirAttentionEncoder = the self-attention encoder of flow_bed (tokens -> attention without positional encoding -> mean,
+ log(1 + n)) with two options:
  y_transform "sqrt": y -> sqrt(max(y, 0) + 1) before the standardisation (the SDE noise scales like sqrt(I), so this
                      evens it out; the +1 keeps d/dy finite at y = 0 for the straight-through design gradient);
  ordered:            the pairs are sorted by their time and each one gets, besides (y, time), the differences
                      (dy, dtime) to its predecessor in time (0 for the first) and a learned embedding of its rank. The
                      posterior only depends on the set of pairs (each carries its time), so this adds no information,
                      but gives the encoder the time structure (growth, peak, decline) directly.
With ``ordered``, ``tokens`` only standardises (the features need the other pairs) and ``pool`` does the rest, so the
tokens / pool interface of the rollouts (flow_bed.sir.builders.rollout_designs) is unchanged.
  attn_activation "silu": SiLU instead of PyTorch's default ReLU in the feed-forward part of the attention layers.
build_sir_flow(cfg) builds the original flow_bed modules for configs without these options (old checkpoints load
unchanged).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from flow_bed.lf.builders import build_encoder
from flow_bed.models import PriorFlowModel
from flow_bed.models.flow_matching import MLP, LikelihoodGuidanceModel
from flow_bed.models.history_encoder import SelfAttentionHistoryEncoder

SPEC = SimpleNamespace(dim=2, eps_dim=1, y_dim=1)
MAX_RANK = 16


class SirAttentionEncoder(SelfAttentionHistoryEncoder):
    def __init__(self, *args, y_transform: str = "linear", ordered: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_transform, self.ordered = y_transform, ordered
        if ordered:
            act = kwargs.get("activation", "silu")
            hidden = list(args[2]) if len(args) > 2 else list(kwargs["hidden_dims"])
            self.phi = MLP(4, hidden, self.embed_dim, act)  # (y, eps, dy, deps)
            self.rank = nn.Embedding(MAX_RANK, self.embed_dim)

    def transform_y(self, y: Tensor) -> Tensor:
        return (y.clamp(min=0.0) + 1.0).sqrt() if self.y_transform == "sqrt" else y

    def tokens(self, y: Tensor, eps: Tensor) -> Tensor:
        ys = (self.transform_y(y) - self.y_loc) / self.y_scale
        if not self.ordered:
            return self.phi(torch.cat([ys, eps], dim=-1))
        raw = torch.cat([ys, eps], dim=-1)  # (B, n, 2), padded to embed_dim to keep the token shape
        return torch.cat([raw, raw.new_zeros(*raw.shape[:-1], self.embed_dim - 2)], dim=-1)

    def pool(self, tokens: Tensor) -> Tensor:
        if not self.ordered:
            return super().pool(tokens)
        B, n, _ = tokens.shape
        if n == 0:
            return torch.cat([tokens.new_zeros(B, self.embed_dim), tokens.new_zeros(B, 1)], dim=-1)
        x = tokens[..., :2]
        order = x[..., 1].argsort(dim=1)
        x = x.gather(1, order[..., None].expand(-1, -1, 2))
        d = x - torch.cat([x[:, :1], x[:, :-1]], dim=1)
        e = self.phi(torch.cat([x, d], dim=-1)) + self.rank(torch.arange(n, device=x.device))[None]
        pooled = self.attention(e).mean(dim=1)
        return torch.cat([pooled, tokens.new_full((B, 1), math.log1p(n))], dim=-1)


def has_sir_options(cfg: dict) -> bool:
    return (bool(cfg.get("ordered_encoder")) or cfg.get("y_transform", "linear") != "linear"
            or cfg.get("attn_activation", "relu") != "relu")


def build_sir_encoder(cfg: dict):
    if not has_sir_options(cfg):
        return build_encoder(cfg, SPEC)
    enc = SirAttentionEncoder(
        SPEC.y_dim, SPEC.eps_dim, cfg["enc_hidden_dims"], cfg["embed_dim"], cfg["attn_layers"], cfg["attn_heads"],
        y_loc=cfg.get("y_loc", 0.0), y_scale=cfg.get("y_scale", 1.0), activation=cfg.get("activation", "silu"),
        ff_dim=cfg.get("attn_ff_dim"), y_transform=cfg.get("y_transform", "linear"),
        ordered=bool(cfg.get("ordered_encoder")),
    )
    if cfg.get("attn_activation", "relu") == "silu":  # same layers as flow_bed's, with SiLU in the feed-forward part
        layer = nn.TransformerEncoderLayer(
            cfg["embed_dim"], cfg["attn_heads"], dim_feedforward=cfg.get("attn_ff_dim") or 2 * cfg["embed_dim"],
            dropout=0.0, batch_first=True, norm_first=True, activation=F.silu,
        )
        enc.attention = nn.TransformerEncoder(layer, cfg["attn_layers"], enable_nested_tensor=False)
    return enc


def build_sir_guidance(cfg: dict, prior: PriorFlowModel) -> LikelihoodGuidanceModel:
    return LikelihoodGuidanceModel(
        prior, build_sir_encoder(cfg), hidden_dims=cfg.get("guidance_hidden_dims") or cfg["hidden_dims"],
        parametrization=cfg.get("param", "w"), activation=cfg.get("activation", "silu"),
    )


def build_sir_flow(cfg: dict) -> tuple[PriorFlowModel, LikelihoodGuidanceModel]:
    prior = PriorFlowModel(SPEC.dim, hidden_dims=cfg["hidden_dims"], activation=cfg.get("activation", "silu"))
    return prior, build_sir_guidance(cfg, prior)
