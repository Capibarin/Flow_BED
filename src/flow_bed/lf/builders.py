"""Location finding: simulator, flow and policy from the config dicts stored in the checkpoints."""

from __future__ import annotations

from flow_bed.models import LikelihoodGuidanceModel, PriorFlowModel
from flow_bed.models.flow_matching import LEGACY_ACTIVATION
from flow_bed.models.history_encoder import HistoryEncoder, SelfAttentionHistoryEncoder
from flow_bed.models.policy import Policy
from flow_bed.simulators.location_finding import LocationFinding

OUTCOME_KEYS = ("sigma", "m")
ENCODER_KEYS = ("encoder", "enc_hidden_dims", "embed_dim", "attn_layers", "attn_heads", "y_loc", "y_scale")


def build_sim(cfg: dict) -> LocationFinding:
    """Simulator with the outcome model (sigma, m) of a config dict; keys that are missing or None keep the defaults
    (checkpoints from before these settings used the defaults)."""
    return LocationFinding(**{k: cfg[k] for k in OUTCOME_KEYS if cfg.get(k) is not None})



def resolve_sim(flow_cfg: dict | None, **given: float | None) -> LocationFinding:
    """Simulator with the outcome model of a flow's config, or without a flow of ``given`` (sigma, m; None = default).
    A value given together with a flow must equal the flow's (ValueError otherwise)."""
    if flow_cfg is None:
        return build_sim(given)
    sim = build_sim(flow_cfg)
    for k, v in given.items():
        if v is not None and v != getattr(sim, k):
            raise ValueError(f"--{k} {v} differs from the flow's ({getattr(sim, k)})")
    return sim

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
    """Prior flow and guidance, e.g. to load a checkpoint of train_lf_flow.py."""
    prior = PriorFlowModel(sim.dim, hidden_dims=cfg["hidden_dims"], activation=cfg.get("activation", LEGACY_ACTIVATION))
    return prior, build_guidance(cfg, sim, prior)


def load_policy(pk: dict, guidance: LikelihoodGuidanceModel, sim: LocationFinding) -> Policy:
    """Policy from a checkpoint of train_lf_policy.py: on its own encoder if it has one, else on the guidance model's."""
    if "encoder" in pk:
        encoder = build_encoder(pk["config"]["policy_encoder"], sim)
        encoder.load_state_dict(pk["encoder"])
    else:
        encoder = guidance.encoder
    policy = Policy(encoder, sim.eps_dim, pk["config"]["policy_hidden_dims"],
                    pk["config"].get("activation", LEGACY_ACTIVATION), pk["config"].get("design_bound"),
                    pk["config"].get("bound_type", "radial"))
    policy.net.load_state_dict(pk["policy"])
    return policy
