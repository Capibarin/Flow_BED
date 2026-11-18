import math

import torch

from flow_bed.models import PosteriorSampler, PriorFlowModel
from flow_bed.models.guarded_sampling import frozen_log_prob
from flow_bed.simulators import sir


def make_bank(ys: torch.Tensor, z: torch.Tensor) -> sir.PathBank:
    """A PathBank around given paths, without simulating (CPU tests)."""
    bank = sir.PathBank.__new__(sir.PathBank)
    bank.n, bank.device = ys.shape[0], ys.device
    bank.gen = torch.Generator().manual_seed(0)
    bank.dt_grid = sir.T_END / (sir.GRID - 1)
    bank.z, bank.ys = z, ys
    bank.valid = (ys.mean(1) >= 1).nonzero()[:, 0]
    return bank


def test_filter_drops_died_out_paths():
    ys = torch.ones(3, sir.GRID)
    ys[1] = 0.1  # mean I < 1: dropped
    bank = make_bank(ys, torch.zeros(3, 2))
    assert bank.valid.tolist() == [0, 2]
    idx = bank.sample_idx(64)
    assert ((idx == 0) | (idx == 2)).all()


def test_observe_value_and_straight_through_gradient():
    slope = 3.0  # I(t) = slope * t: the central difference is exact for any window
    bank = make_bank(torch.arange(sir.GRID, dtype=torch.float32)[None] * (sir.T_END / (sir.GRID - 1)) * slope,
                     torch.zeros(1, 2))
    for w in (1, 50):
        tau = torch.tensor([[37.3]], requires_grad=True)
        y = bank.observe(torch.tensor([0]), tau, w=w)
        k = round(37.3 / bank.dt_grid)
        assert y.shape == (1, 1, 1)
        assert math.isclose(y.item(), slope * k * bank.dt_grid, rel_tol=1e-4)
        (g,) = torch.autograd.grad(y.sum(), tau)
        assert math.isclose(g.item(), slope, rel_tol=1e-3)  # float32 central difference of the ramp


def test_design_transforms():
    prev = torch.full((8, 1), sir.LB)
    tau = sir.next_tau(torch.randn(8, 1), prev)
    assert (tau > prev).all() and (tau < sir.UB).all()
    eq = sir.equal_interval_taus(5)
    assert eq.shape == (5,) and (eq.diff() > 0).all() and eq.min() > 0 and eq.max() < sir.T_END
    g = torch.Generator().manual_seed(0)
    rnd = sir.random_taus(16, 5, g, torch.device("cpu"))
    uni = sir.uniform_taus(16, 5, g, torch.device("cpu"))
    for t in (rnd, uni):
        assert t.shape == (16, 5) and t.min() >= sir.LB and t.max() <= sir.UB
    z = torch.zeros(4, 2)
    assert torch.allclose(sir.z_to_theta(z), torch.tensor([0.5, 0.1]).expand(4, -1))


def test_freeze_guard_isolates_far_points():
    torch.manual_seed(0)
    prior = PriorFlowModel(2, hidden_dims=[32, 32])
    sampler = PosteriorSampler(prior, None, solver="dopri5", divergence="exact")
    theta = torch.tensor([[0.1, -0.2], [80.0, 80.0]])  # the second point starts outside the ball |x| <= 20
    lq, frozen, calls = frozen_log_prob(sampler, theta, None, R=20.0, cap=3000)
    assert frozen.tolist() == [False, True]
    assert torch.isfinite(lq[0])
    assert lq[1] == -float("inf")  # frozen: density 0
    assert calls <= 3000
