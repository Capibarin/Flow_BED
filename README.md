# Flow_BED

Flow matching with learned likelihood guidance for Bayesian experimental design.

```
src/flow_bed/
  config.py              YAML configs for the training scripts (--config)
  models/                PriorFlowModel, LikelihoodGuidanceModel, history encoders, Policy, PosteriorSampler
  simulators/            linear_gaussian.py (toy problem with analytic posterior), location_finding.py
  lf/                    location finding: builders.py (simulator, flow and policy from checkpoint configs),
                         training.py, rollouts.py, true_posterior.py
runs/
  configs/lf/            prior.yaml, flow_attn.yaml (encoder + guidance), policy_div.yaml, joint_div.yaml
  sbatches/              lf_pipeline.sbatch: prior -> guidance -> policy on one GPU (SLURM)
  scripts/
    training/            train_lf_flow.py, train_lf_policy.py, train_lf_joint.py, train_toy.py
    evaluation/          eval_lf_spce.py, eval_lf_eig_per_step.py, eval_lf_posterior.py, eval_toy_nobs.py
    analysis/            tau_rms.py (design-gradient RMS per flow time)
    figures/             lf_rollout_figures.py, lf_true_posterior_figures.py, plot_rho.py, sample_toy.py
tests/
```

## Setup

```bash
uv venv && source .venv/bin/activate      # or python -m venv .venv
uv pip install -e .                       # or pip install -e .; ".[dev]" adds pytest and ruff
pytest
python runs/scripts/training/train_toy.py && python runs/scripts/figures/sample_toy.py
```

`pip install -e .` installs everything the scripts need. Plain pip pulls the default CUDA build of torch; if the
machine's driver is older, install torch for its CUDA version first (e.g. `pip install torch --index-url
https://download.pytorch.org/whl/cu126`).

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

Three training scripts, each configured by a YAML file in `runs/configs/lf/` (`--config`; any argument given on the
command line overrides the file, see `src/flow_bed/config.py`). The configs reproduce the current runs: the 400k-step
attention posterior, the decoupled divergence policy and the joint policy + posterior training.

The outcome model `y ~ N(log(b + sum_k a / (m + |xi - theta_k|^2)), sigma^2)` is set when the flow is trained
(`sigma`, `m` in `flow_attn.yaml`, or `--sigma`, `--m`) and stored in its checkpoint; the policy, evaluation and figure
scripts take it from the flow checkpoint. Checkpoints without these keys use the defaults (0.5, 1e-4). Scripts that
can run on random designs alone (`eval_lf_spce.py`, `eval_lf_eig_per_step.py`, `lf_true_posterior_figures.py`)
take `--sigma` and `--m` for that case; next to a flow, a value that differs from the flow's is an error.

```bash
# 1) posterior flow on random designs, from scratch: prior (50 x 1000 steps), then encoder + guidance (400 x 1000 steps)
python runs/scripts/training/train_lf_flow.py --config runs/configs/lf/flow_attn.yaml --out checkpoints/lf_flow_attn.pt
#    or with the frozen prior of an existing flow checkpoint (encoder + guidance only, ~3 h); with prior_silu.pt this
#    reproduces checkpoints/train_curve_anneal_silu/ckpts/s1_400k_attn.pt
python runs/scripts/training/train_lf_flow.py --config runs/configs/lf/flow_attn.yaml \
    --out checkpoints/lf_flow_attn.pt --init-prior checkpoints/train_curve_anneal_silu/ckpts/prior_silu.pt

# 2a) decoupled: policy on the frozen flow, 400 x 781 steps (~2 days)
python runs/scripts/training/train_lf_policy.py --config runs/configs/lf/policy_div.yaml \
    --flow-ckpt checkpoints/lf_flow_attn.pt --policy-term final_random --out checkpoints/lf_policy_div_finalrandq.pt \
    --log-csv checkpoints/logs/lf_policy_div_finalrandq.csv

# 2b) or joint: policy and posterior (encoder + guidance) together, starting from the flow, 400 x 781 steps (~2 days)
python runs/scripts/training/train_lf_joint.py --config runs/configs/lf/joint_div.yaml \
    --init-flow checkpoints/lf_flow_attn.pt --policy-term final_random --out checkpoints/lf_joint_div_finalrandq \
    --log-csv checkpoints/logs/lf_joint_div_finalrandq.csv

# 3) evaluation: sPCE (T = 30) and the Figure 5 rollouts (joint: the flow is <out>_flow.pt, the policy <out>_policy.pt)
python runs/scripts/evaluation/eval_lf_spce.py --flow-ckpt checkpoints/lf_flow_attn.pt \
    --policy finalrandq=checkpoints/lf_policy_div_finalrandq.pt
python runs/scripts/figures/lf_rollout_figures.py --flow-ckpt checkpoints/lf_flow_attn.pt \
    --policy finalrandq=checkpoints/lf_policy_div_finalrandq.pt --out checkpoints/figures_v2
```

`--init-prior` can point to any flow checkpoint (or a prior-only one with "config" and "prior"). Always pass `--out`:
the configs do not set it, and the scripts' default paths may hold older checkpoints. `--ckpt-every n` saves the
current model to `--out` every n epochs.

Policy variants are flags on top of `policy_div.yaml`: `--detach-policy-input` (no gradient through the history the
policy sees), `--greedy` (each step's term trains only its own design), `--policy-term final_random`.

### On a SLURM cluster

`runs/sbatches/lf_pipeline.sbatch` trains the prior (`prior.yaml`), then encoder + guidance on it (`flow_attn.yaml`),
then a policy (`policy_div.yaml`) into `checkpoints/runs/$RUN`; `runs/sbatches/lf_eval.sbatch` evaluates a policy
(sPCE and Figures 2 and 5). Settings are environment variables, see the top of each file.

```bash
mkdir -p logs
RUN=sigma1 FLOW_ARGS="--sigma 1.0" POLICY=detach POLICY_ARGS="--detach-policy-input" \
    sbatch runs/sbatches/lf_pipeline.sbatch
RUN=sigma1 POLICY=detach sbatch --dependency=afterok:<training job id> runs/sbatches/lf_eval.sbatch
```

### Flow-time density of the policy gradient

```bash
# design-gradient RMS per flow time tau on the frozen flow (random and policy designs), then rho ∝ RMS
python runs/scripts/analysis/tau_rms.py --flow-ckpt checkpoints/lf_flow_attn.pt \
    --policy checkpoints/lf_policy_div_finalrandq.pt --out checkpoints/tau_rms.json
python runs/scripts/figures/plot_rho.py --rms checkpoints/tau_rms.json --out checkpoints/figures_v2/tau_rho.png
```
