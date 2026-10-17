import math

import pytest
import torch

from flow_bed.models import LikelihoodGuidanceModel, PosteriorSampler, PriorFlowModel
from flow_bed.models.history_encoder import HistoryEncoder
from flow_bed.models.posterior_sampler import exact_divergence, hutchinson_divergence

D, DY, DE = 3, 1, 3
HIDDEN = [32, 32]
# (solver kwargs, tolerance of the forward/backward log-density check)
# torchdyn's RungeKutta4 evaluates its stages at the wrong times (t, t + h/2, t + h/2 instead of t + h/2, t + h/2,
# t + h), which makes it first order for time-dependent fields, hence the many steps and the looser tolerance.
SOLVERS = [
    pytest.param({"solver": "dopri5", "atol": 1e-6, "rtol": 1e-6}, 1e-3, id="dopri5"),
    pytest.param({"solver": "rk4", "n_steps": 500}, 1e-2, id="rk4", marks=pytest.mark.slow),
]


@pytest.fixture
def models():
    torch.manual_seed(0)
    prior = PriorFlowModel(D, hidden_dims=HIDDEN)
    return prior, LikelihoodGuidanceModel(prior, HistoryEncoder(DY, DE, hidden_dims=HIDDEN), hidden_dims=HIDDEN)


@pytest.mark.slow
def test_prior_shapes_and_loss(models):
    prior, _ = models
    theta = torch.randn(8, D)
    v, s = prior(theta, torch.rand(8))
    assert v.shape == s.shape == (8, D)
    assert prior.shared_step(theta).ndim == 0


def test_lambda_sp_is_lambda_times_derivative():
    prior = PriorFlowModel(D, hidden_dims=HIDDEN, sigma_min=0.1, sigma=0.3)
    t = torch.rand(10, requires_grad=True)
    lam = prior.lambda_t(t)
    (dlam,) = torch.autograd.grad(lam.sum(), t)
    assert torch.allclose(prior.lambda_sp_t(t), lam * dlam, atol=1e-6)


def test_guidance(models):
    _, g = models
    theta, t = torch.randn(5, D), torch.full((5,), 0.7)
    y, eps = torch.randn(5, 4, DY), torch.randn(5, 4, DE)
    perm = torch.randperm(4)
    w = g(theta, t, g.encoder(y, eps))
    assert torch.allclose(w, g(theta, t, g.encoder(y[:, perm], eps[:, perm])), atol=1e-5)  # order invariant
    assert g.shared_step((theta, eps, y)).ndim == 0
    assert all(not p.requires_grad for p in g.prior.parameters())


@pytest.mark.slow
def test_divergence_estimators_agree_on_linear_field():
    A = torch.randn(D, D)
    x = torch.randn(6, D, requires_grad=True)
    v = x @ A.T
    div = exact_divergence(v, x, create_graph=False)
    assert torch.allclose(div, torch.full((6,), torch.trace(A).item()), atol=1e-5)
    h = torch.stack([hutchinson_divergence(v, x, torch.randn_like(x), create_graph=False) for _ in range(2000)])
    assert h.shape[1:] == (6,)
    assert torch.allclose(h.mean(0), div, atol=0.3)  # unbiased estimator


@pytest.mark.parametrize("solver_kw,tol", SOLVERS)
@pytest.mark.parametrize("divergence", ["exact", pytest.param("hutchinson", marks=pytest.mark.slow)])
def test_sampler_roundtrip(models, solver_kw, tol, divergence):
    prior, g = models
    sampler = PosteriorSampler(prior, g, divergence=divergence, **solver_kw)
    with torch.no_grad():
        history = g.encoder(torch.randn(2, 3, DY), torch.randn(2, 3, DE))
        theta, logq = sampler.sample(5, history, return_log_prob=True)
        assert theta.shape == (2, 5, D) and logq.shape == (2, 5)
        lp = sampler.log_prob(theta.reshape(-1, D), history.repeat_interleave(5, 0))
    assert torch.isfinite(lp).all() and lp.shape == (10,)
    uncond = sampler.sample(7)
    assert uncond.shape == (7, D)


@pytest.mark.parametrize("solver_kw,tol", SOLVERS)
def test_logprob_forward_backward_agree(models, solver_kw, tol):
    """log q from the forward (sampling) pass must match log_prob from the reversed pass."""
    prior, g = models
    sampler = PosteriorSampler(prior, g, divergence="exact", **solver_kw)
    with torch.no_grad():
        history = g.encoder(torch.randn(1, 3, DY), torch.randn(1, 3, DE))
        theta, logq_fwd = sampler.sample(16, history, return_log_prob=True)
        logq_bwd = sampler.log_prob(theta[0], history.expand(16, -1))
    assert torch.allclose(logq_fwd[0], logq_bwd, atol=tol)


@pytest.mark.parametrize("solver_kw,tol", SOLVERS)
def test_identity_flow_logprob_is_gaussian(solver_kw, tol):
    """A zero velocity field leaves the base Gaussian untouched: log q(theta) == log N(theta)."""

    class Zero(PriorFlowModel):
        def forward(self, theta_t, t):
            v, s = super().forward(theta_t, t)
            return 0.0 * v, s

    sampler = PosteriorSampler(Zero(D, hidden_dims=[8]), **solver_kw)
    theta = torch.randn(10, D)
    lp = sampler.log_prob(theta)
    ref = -0.5 * (theta**2).sum(-1) - 0.5 * D * math.log(2 * math.pi)
    assert torch.allclose(lp, ref, atol=1e-5)
