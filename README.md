# Flow_BED

Flow matching with learned likelihood guidance for Bayesian experimental design.

```
src/flow_bed/
  models/
    flow_matching.py   PriorFlowModel (p(theta)), LikelihoodGuidanceModel (grad_theta log p(y | theta, eps))
    nodewrappers.py    torchdyn wrappers: GuidedVectorField, CNFVectorField, PosteriorSampler
  simulators/
    linear_gaussian.py toy problem with analytic posterior
scripts/
  train_toy.py         train prior + guidance on the toy problem
  sample_toy.py        guided posterior sampling + log-likelihood, compared to the analytic posterior
tests/
```

## Setup

```bash
uv venv && source .venv/bin/activate      # or python -m venv .venv
uv pip install -e ".[dev]"                # or pip install -e ".[dev]"
pytest
python scripts/train_toy.py && python scripts/sample_toy.py
```

## Idea

Linear path `theta_t = t theta_1 + (1 - t) theta_0`, `theta_0 ~ N(0, I)`. For this path the score and the
velocity are related by `v_t(x) = x / t + c(t) s_t(x)` with `c(t) = sigma_t / t`, so Bayes' rule in score
space becomes

```
v_t(x | y, eps) = v_prior(x, t) + c(t) * grad_x log p_t(y | x, eps)
```

`PriorFlowModel` learns `v_prior`; `LikelihoodGuidanceModel` learns the correction `g ≈ grad_x log p_t(y | x, eps)`
by regressing `v_prior + c(t) g` onto the conditional flow-matching target on joint simulator samples
`(theta, eps, y)`. For several observations the term factorises, `sum_i g(x, t, y_i, eps_i)`, which the model
does internally (with an optional mask for variable numbers of observations).

`PosteriorSampler` integrates the guided field with torchdyn (`sample`, 0 → 1) and evaluates
`log q(theta | y, eps)` by integrating the reversed field with the divergence term (`log_prob`, 1 → 0),
using either the exact trace or a Hutchinson estimator.

## Location finding

Three training scripts, each configured by a YAML file in `configs/lf/` (`--config`; any argument given on the command
line overrides the file, see `src/flow_bed/config.py`). The configs reproduce the current runs: the 400k-step attention
posterior, the decoupled divergence policy and the joint policy + posterior training.

```bash
# 1) posterior flow on random designs, from scratch: prior (50 x 1000 steps), then encoder + guidance (400 x 1000 steps)
python scripts/train_lf_flow.py --config configs/lf/flow_attn.yaml --out checkpoints/lf_flow_attn.pt
#    or with the frozen prior of an existing flow checkpoint (encoder + guidance only, ~3 h); with prior_silu.pt this
#    reproduces checkpoints/train_curve_anneal_silu/ckpts/s1_400k_attn.pt
python scripts/train_lf_flow.py --config configs/lf/flow_attn.yaml --out checkpoints/lf_flow_attn.pt \
    --init-prior checkpoints/train_curve_anneal_silu/ckpts/prior_silu.pt

# 2a) decoupled: policy on the frozen flow, 400 x 781 steps (~2 days)
python scripts/train_lf_policy.py --config configs/lf/policy_div.yaml --flow-ckpt checkpoints/lf_flow_attn.pt \
    --policy-term final_random --out checkpoints/lf_policy_div_finalrandq.pt \
    --log-csv checkpoints/logs/lf_policy_div_finalrandq.csv

# 2b) or joint: policy and posterior (encoder + guidance) together, starting from the flow, 400 x 781 steps (~2 days)
python scripts/train_lf_joint.py --config configs/lf/joint_div.yaml --init-flow checkpoints/lf_flow_attn.pt \
    --policy-term final_random --out checkpoints/lf_joint_div_finalrandq \
    --log-csv checkpoints/logs/lf_joint_div_finalrandq.csv

# 3) evaluation: sPCE (T = 30) and the Figure 5 rollouts (joint: the flow is <out>_flow.pt, the policy <out>_policy.pt)
python scripts/eval_lf_spce.py --flow-ckpt checkpoints/lf_flow_attn.pt \
    --policy finalrandq=checkpoints/lf_policy_div_finalrandq.pt
python scripts/lf_rollout_figures.py --flow-ckpt checkpoints/lf_flow_attn.pt \
    --policy finalrandq=checkpoints/lf_policy_div_finalrandq.pt --out checkpoints/figures_v2
```

`--init-prior` can point to any flow checkpoint (or a prior-only one with "config" and "prior"). Always pass `--out`:
the configs do not set it, and the scripts' default paths may hold older checkpoints. `--ckpt-every n` saves the
current model to `--out` every n epochs.
