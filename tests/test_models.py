import math

import pytest
import torch

from flow_bed.models import CNFVectorField, GuidedVectorField, LikelihoodGuidanceModel, PosteriorSampler, PriorFlowModel
from flow_bed.models.nodewrappers import exact_divergence, hutchinson_divergence

D, DY, DE = 3, 1, 3


@pytest.fixture
def models():
    torch.manual_seed(0)
    return PriorFlowModel(D, hidden=32, n_layers=2), LikelihoodGuidanceModel(D, DY, DE, hidden=32, n_layers=2)


def test_prior_loss_and_shapes(models):
    prior, _ = models
    theta = torch.randn(8, D)
    assert prior(theta, 0.5).shape == (8, D)
    assert prior(theta, torch.rand(8)).shape == (8, D)
    assert prior.loss(theta).ndim == 0


def test_guidance_sums_over_observations(models):
    prior, g = models
    theta = torch.randn(5, D)
    y, eps = torch.randn(5, 4, DY), torch.randn(5, 4, DE)
    total = g(theta, 0.7, y, eps)
    per = torch.stack([g(theta, 0.7, y[:, i], eps[:, i]) for i in range(4)]).sum(0)
    assert torch.allclose(total, per, atol=1e-5)
    mask = torch.tensor([[1, 1, 0, 0]] * 5, dtype=torch.bool)
    masked = g(theta, 0.7, y, eps, mask)
    assert torch.allclose(masked, g(theta, 0.7, y[:, :2], eps[:, :2]), atol=1e-5)
    assert g.loss(prior, theta, y, eps).ndim == 0


def test_divergence_estimators_agree_on_linear_field():
    A = torch.randn(D, D)
    x = torch.randn(6, D, requires_grad=True)
    v = x @ A.T
    div = exact_divergence(v, x, create_graph=False)
    assert torch.allclose(div, torch.full((6,), torch.trace(A).item()), atol=1e-5)
    h = torch.stack([hutchinson_divergence(v, x, torch.randn_like(x), create_graph=False) for _ in range(2000)])
    assert h.shape[1:] == (6,)
    assert torch.allclose(h.mean(0), div, atol=0.3)  # unbiased estimator


def test_cnf_reverse_sign(models):
    prior, g = models
    vf = GuidedVectorField(prior, g)
    vf.set_context(torch.randn(4, 2, DY), torch.randn(4, 2, DE))
    fwd, bwd = CNFVectorField(vf, "exact"), CNFVectorField(vf, "exact", reverse=True)
    s = torch.cat([torch.randn(4, D), torch.zeros(4, 1)], -1)
    t = torch.tensor(0.3)
    assert torch.allclose(fwd(t, s), -bwd(1 - t, s), atol=1e-6)


@pytest.mark.parametrize("divergence", ["exact", "hutchinson"])
def test_sampler_roundtrip(models, divergence):
    prior, g = models
    sampler = PosteriorSampler(prior, g, solver="rk4", n_steps=20, divergence=divergence)
    y, eps = torch.randn(2, 3, DY), torch.randn(2, 3, DE)
    with torch.no_grad():
        theta, logq = sampler.sample(5, y=y, eps=eps, return_log_prob=True)
        assert theta.shape == (2, 5, D) and logq.shape == (2, 5)
        lp = sampler.log_prob(theta.reshape(-1, D), y=y.repeat_interleave(5, 0), eps=eps.repeat_interleave(5, 0))
    assert torch.isfinite(lp).all() and lp.shape == (10,)
    uncond = sampler.sample(7)
    assert uncond.shape == (7, D)


def test_prior_logprob_forward_backward_agree(models):
    """log q from the forward (sampling) pass must match log_prob from the reversed pass."""
    prior, _ = models
    sampler = PosteriorSampler(prior, solver="rk4", n_steps=50, divergence="exact")
    with torch.no_grad():
        theta, logq_fwd = sampler.sample(16, return_log_prob=True)
        logq_bwd = sampler.log_prob(theta)
    assert torch.allclose(logq_fwd, logq_bwd, atol=1e-3)


def test_identity_flow_logprob_is_gaussian():
    """A zero velocity field leaves the base Gaussian untouched: log q(theta) == log N(theta)."""

    class Zero(PriorFlowModel):
        def forward(self, theta_t, t):
            return torch.zeros_like(theta_t) + 0.0 * super().forward(theta_t, t)

    sampler = PosteriorSampler(Zero(D, hidden=8, n_layers=2), solver="rk4", n_steps=5)
    theta = torch.randn(10, D)
    lp = sampler.log_prob(theta)
    ref = -0.5 * (theta**2).sum(-1) - 0.5 * D * math.log(2 * math.pi)
    assert torch.allclose(lp, ref, atol=1e-5)
