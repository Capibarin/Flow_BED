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
