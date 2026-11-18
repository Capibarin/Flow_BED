"""SDE-based SIR model with the iDAD setup (Ivanova et al. 2021, epidemic_simulate_data.py / epidemic.py).

theta = (beta, gamma), log theta ~ N(log(0.5, 0.1), 0.5^2 I). The flow works on z = (log theta - log(0.5, 0.1)) / 0.5,
so the prior on z is N(0, I) (before the filter below).
X = (S, I), N = 500, X(0) = (498, 2), Ito SDE dX = f dt + G dW with
    p_inf = beta S I / N,  p_rec = gamma I,  f = (-p_inf, p_inf - p_rec),
    dS = -p_inf dt - sqrt(p_inf) dW1,  dI = (p_inf - p_rec) dt + sqrt(p_inf) dW1 - sqrt(p_rec) dW2,
Euler-Maruyama with the state clamped to [0, N] before every step (as iDAD / torchsde with dt = 1e-3), I recorded on the
grid ts = linspace(0, 100, 10000). Paths with mean I < 1 over the grid (epidemics that die out at once) are dropped,
as in iDAD, so the effective prior is the filtered one.
Observation y = I(tau) at the nearest interior grid point (ts[1:-1]); d y / d tau = central difference of the path
(iDAD's SIR_SDE_Simulator), implemented as a straight-through term.
Designs: increasing measurement times tau_t = tau_{t-1} + sigmoid(u_t) (UB - tau_{t-1}), tau_0 = LB (iDAD's "ts"
transform), LB = 0.01, UB = 99.99. The history stores the time as eps = tau / 50 - 1 in [-1, 1].
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

LOC = torch.tensor([math.log(0.5), math.log(0.1)])
SCALE = 0.5
N_POP, I0 = 500.0, 2.0
T_END, GRID = 100.0, 10000
LB, UB = 1e-2, 100.0 - 1e-2
SUBSTEPS = 10  # Euler steps per grid interval: dt = 100 / 9999 / 10 ~ 1e-3


def z_to_theta(z: Tensor) -> Tensor:
    return (LOC.to(z.device) + SCALE * z).exp()


def tau_to_eps(tau: Tensor) -> Tensor:
    return tau / 50.0 - 1.0


@torch.no_grad()
def simulate_paths(n: int, gen: torch.Generator, device, chunk: int = 50_000,
                   z_given: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """n prior draws (or the given z_given (n, 2)) -> z (n, 2), I on the grid (n, GRID) float32, and the indices of
    the paths that pass the mean-I >= 1 filter (kept in place, without a filtered copy, to save memory)."""
    dt_grid = T_END / (GRID - 1)
    dt = dt_grid / SUBSTEPS
    zs = torch.empty(n, 2, device=device)
    ys = torch.empty(n, GRID, device=device)
    for start in range(0, n, chunk):
        b = min(chunk, n - start)
        z = torch.randn(b, 2, device=device, generator=gen) if z_given is None else z_given[start:start + b]
        beta, gamma = z_to_theta(z).unbind(-1)
        S = torch.full((b,), N_POP - I0, device=device)
        I = torch.full((b,), I0, device=device)
        out = ys[start:start + b]
        out[:, 0] = I
        sq = math.sqrt(dt)
        for k in range(1, GRID):
            for _ in range(SUBSTEPS):
                S.clamp_(0.0, N_POP)
                I.clamp_(0.0, N_POP)
                p_inf = beta * S * I / N_POP
                p_rec = gamma * I
                dw = torch.randn(2, b, device=device, generator=gen) * sq
                a, r = p_inf.sqrt() * dw[0], p_rec.sqrt() * dw[1]
                S, I = S - p_inf * dt - a, I + (p_inf - p_rec) * dt + a - r
            out[:, k] = I
        zs[start:start + b] = z
    return zs, ys, (ys.mean(1) >= 1).nonzero()[:, 0]


class PathBank:
    """Pre-simulated paths on the GPU; ``refresh`` re-simulates them (iDAD: every 10k steps)."""

    def __init__(self, n: int, seed: int, device):
        self.n, self.device = n, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.dt_grid = T_END / (GRID - 1)
        self.refresh()

    def refresh(self) -> None:
        self.z = self.ys = self.valid = None
        torch.cuda.empty_cache()
        self.z, self.ys, self.valid = simulate_paths(self.n, self.gen, self.device)

    def sample_idx(self, b: int) -> Tensor:
        """Indices of b paths drawn uniformly from those that pass the filter."""
        return self.valid[torch.randint(0, self.valid.shape[0], (b,), device=self.device, generator=self.gen)]

    def observe(self, idx: Tensor, tau: Tensor, w: int = 1) -> Tensor:
        """idx (B,), tau (B, n) -> y = I(tau) (B, n, 1); d y / d tau = central difference over +-w grid steps
        (straight-through; w = 1: one grid step each side; larger w smooths the rough path's derivative)."""
        with torch.no_grad():
            k = (tau / self.dt_grid).round().long().clamp(1, GRID - 2)
            kp, km = (k + w).clamp(max=GRID - 1), (k - w).clamp(min=0)
            rows = idx[:, None].expand_as(k)
            y = self.ys[rows, k]
            dy = (self.ys[rows, kp] - self.ys[rows, km]) / ((kp - km) * self.dt_grid)
        if tau.requires_grad:
            y = y + (tau - tau.detach()) * dy
        return y[..., None]


def next_tau(u: Tensor, tau_prev: Tensor) -> Tensor:
    """iDAD's increasing transform: tau = tau_prev + sigmoid(u) (UB - tau_prev)."""
    return tau_prev + torch.sigmoid(u) * (UB - tau_prev)


def equal_interval_taus(T: int) -> Tensor:
    """iDAD's equal-interval baseline: LB + linspace(0.01, 0.99, T) (UB - LB)."""
    return LB + torch.linspace(0.01, 0.99, T) * (UB - LB)


def random_taus(b: int, T: int, gen: torch.Generator, device) -> Tensor:
    """iDAD's random baseline: independent LB + sigmoid(U(-5, 5)) (UB - LB)."""
    u = torch.rand(b, T, device=device, generator=gen) * 10 - 5
    return LB + torch.sigmoid(u) * (UB - LB)


def uniform_taus(b: int, n: int, gen: torch.Generator, device) -> Tensor:
    """Random designs for the flow's training: independent U(LB, UB)."""
    return LB + torch.rand(b, n, device=device, generator=gen) * (UB - LB)
