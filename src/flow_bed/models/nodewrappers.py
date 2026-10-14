"""torchdyn wrappers for guided posterior sampling and log-likelihood evaluation.

Pieces
------
``GuidedVectorField``   v(t, x) = v_prior(x, t) + lambda(t) * c(t) * sum_i g(x, t, y_i, eps_i)
``CNFVectorField``      augments the state with the log-density:  d/dt [x, logp] = [v, -div v]
``PosteriorSampler``    high-level: ``sample(...)`` (0 -> 1) and ``log_prob(...)`` (1 -> 0) through a
                        torchdyn ``NeuralODE`` (falls back to a built-in fixed-step RK4 if torchdyn is absent).

All vector fields use the torchdyn signature ``forward(t, x, args=None)``.

The log-density returned is that of the *guided flow's* sampling distribution q(theta | y, eps), which
approximates the posterior. With ``guidance=None`` (or ``y=None``) it is the learned prior p(theta).
"""

from __future__ import annotations

from math import log, pi
from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn

from flow_bed.models.flow_matching import LikelihoodGuidanceModel, PriorFlowModel, _as_batch_time

try:  # torchdyn is optional at import time so the module stays testable without it
    from torchdyn.core import NeuralODE

    _HAS_TORCHDYN = True
except Exception:  # pragma: no cover
    NeuralODE = None
    _HAS_TORCHDYN = False


ScaleFn = Union[float, Callable[[Tensor], Tensor]]


# --------------------------------------------------------------------------------------
# Divergence estimators
# --------------------------------------------------------------------------------------


def exact_divergence(v: Tensor, x: Tensor, create_graph: bool) -> Tensor:
    """Exact trace of dv/dx via d autograd calls. ``v``, ``x``: (B, d) -> (B,)."""
    div = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for i in range(x.shape[1]):
        (grad,) = torch.autograd.grad(v[:, i].sum(), x, create_graph=create_graph, retain_graph=True)
        div = div + grad[:, i]
    return div


def hutchinson_divergence(v: Tensor, x: Tensor, noise: Tensor, create_graph: bool) -> Tensor:
    """Hutchinson estimator  E[e^T (dv/dx) e]  with a single probe ``noise`` (B, d) -> (B,)."""
    (vjp,) = torch.autograd.grad(v, x, grad_outputs=noise, create_graph=create_graph, retain_graph=True)
    return (vjp * noise).sum(-1)


# --------------------------------------------------------------------------------------
# Vector fields
# --------------------------------------------------------------------------------------


class GuidedVectorField(nn.Module):
    """Prior velocity plus (optionally) the summed likelihood guidance for the current context.

    Call ``set_context(y, eps, mask)`` with tensors whose batch axis matches the state before integrating;
    ``set_context(None)`` gives the unconditional prior flow.
    """

    def __init__(
        self,
        prior: PriorFlowModel,
        guidance: Optional[LikelihoodGuidanceModel] = None,
        guidance_scale: ScaleFn = 1.0,
        t_clip: float = 1e-3,
    ):
        super().__init__()
        self.prior = prior
        self.guidance = guidance
        self.guidance_scale = guidance_scale
        self.t_clip = t_clip
        self._y: Optional[Tensor] = None
        self._eps: Optional[Tensor] = None
        self._mask: Optional[Tensor] = None

    # context -------------------------------------------------------------------------
    def set_context(self, y: Optional[Tensor], eps: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> None:
        self._y, self._eps, self._mask = y, eps, mask

    @property
    def conditional(self) -> bool:
        return self.guidance is not None and self._y is not None

    def _scale(self, t: Tensor) -> Tensor:
        if callable(self.guidance_scale):
            return self.guidance_scale(t)
        return torch.as_tensor(self.guidance_scale, device=t.device, dtype=t.dtype)

    # torchdyn signature --------------------------------------------------------------
    def forward(self, t: Tensor, x: Tensor, args=None) -> Tensor:
        tb = _as_batch_time(t, x.shape[0], x).clamp(self.t_clip, 1.0)
        v = self.prior(x, tb)
        if self.conditional:
            corr = self.guidance.velocity_correction(x, tb, self._y, self._eps, self._mask)
            v = v + self._scale(tb).reshape(-1, 1) * corr
        return v


class CNFVectorField(nn.Module):
    """Continuous normalising flow augmentation of a vector field.

    State is ``[x, logp]`` of shape (B, d + 1); ``forward`` returns ``[v(t, x), -div v(t, x)]``.
    With ``reverse=True`` the field is ``-v(1 - s, x)`` so that integrating s: 0 -> 1 runs the flow backwards
    in time (used for ``log_prob``) with any solver, including adaptive ones.
    """

    def __init__(
        self,
        vf: nn.Module,
        divergence: str = "exact",
        hutchinson_dist: str = "rademacher",
        reverse: bool = False,
    ):
        super().__init__()
        if divergence not in ("exact", "hutchinson"):
            raise ValueError("divergence must be 'exact' or 'hutchinson'")
        self.vf = vf
        self.divergence = divergence
        self.hutchinson_dist = hutchinson_dist
        self.reverse = reverse
        self._noise: Optional[Tensor] = None

    def reset_noise(self, x: Tensor) -> None:
        """Draw (and fix for the whole trajectory) the Hutchinson probe."""
        if self.hutchinson_dist == "rademacher":
            self._noise = torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2 - 1
        else:
            self._noise = torch.randn_like(x)

    def forward(self, t: Tensor, state: Tensor, args=None) -> Tensor:
        x = state[:, :-1]
        create_graph = torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in self.vf.parameters()))
        with torch.enable_grad():  # torchdyn's adjoint runs the forward under no_grad
            x = x.detach().requires_grad_(True) if not x.requires_grad else x
            t_eff = (1.0 - t) if self.reverse else t
            v = self.vf(t_eff, x)
            if self.divergence == "exact":
                div = exact_divergence(v, x, create_graph)
            else:
                if self._noise is None or self._noise.shape != x.shape:
                    self.reset_noise(x)
                div = hutchinson_divergence(v, x, self._noise, create_graph)
        if not create_graph:
            v, div = v.detach(), div.detach()
        if self.reverse:
            v, div = -v, -div
        return torch.cat([v, -div[:, None]], dim=-1)


# --------------------------------------------------------------------------------------
# Fallback integrator
# --------------------------------------------------------------------------------------


def _rk4(f: Callable[[Tensor, Tensor], Tensor], x: Tensor, t_span: Tensor) -> Tensor:
    for t0, t1 in zip(t_span[:-1], t_span[1:]):
        h = t1 - t0
        k1 = f(t0, x)
        k2 = f(t0 + h / 2, x + h / 2 * k1)
        k3 = f(t0 + h / 2, x + h / 2 * k2)
        k4 = f(t1, x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# --------------------------------------------------------------------------------------
# High-level wrapper
# --------------------------------------------------------------------------------------


class PosteriorSampler(nn.Module):
    """Sample theta ~ q(theta | y, eps) and evaluate log q(theta | y, eps) with torchdyn.

    Parameters
    ----------
    prior, guidance : trained models (``guidance=None`` -> prior only)
    solver          : any torchdyn solver name ('dopri5', 'tsit5', 'rk4', 'euler', ...)
    sensitivity     : 'autograd' or 'adjoint' (torchdyn)
    divergence      : 'exact' (d autograd passes, fine for low-dim theta) or 'hutchinson'
    guidance_scale  : float or callable(t) -> Tensor; multiplies the guidance term
    n_steps         : number of grid points for fixed-step solvers (ignored by adaptive ones)
    """

    def __init__(
        self,
        prior: PriorFlowModel,
        guidance: Optional[LikelihoodGuidanceModel] = None,
        solver: str = "dopri5",
        sensitivity: str = "autograd",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        divergence: str = "exact",
        hutchinson_dist: str = "rademacher",
        guidance_scale: ScaleFn = 1.0,
        t_clip: float = 1e-3,
        n_steps: int = 100,
    ):
        super().__init__()
        self.vf = GuidedVectorField(prior, guidance, guidance_scale, t_clip)
        self.cnf_fwd = CNFVectorField(self.vf, divergence, hutchinson_dist, reverse=False)
        self.cnf_bwd = CNFVectorField(self.vf, divergence, hutchinson_dist, reverse=True)
        self.solver, self.sensitivity, self.atol, self.rtol = solver, sensitivity, atol, rtol
        self.n_steps = n_steps
        self._nodes: dict[str, nn.Module] = {}

    # ---- properties -------------------------------------------------------------------
    @property
    def prior(self) -> PriorFlowModel:
        return self.vf.prior

    @property
    def guidance(self) -> Optional[LikelihoodGuidanceModel]:
        return self.vf.guidance

    @property
    def dim(self) -> int:
        return self.prior.dim

    # ---- integration ------------------------------------------------------------------
    def _t_span(self, ref: Tensor) -> Tensor:
        if self.solver in ("dopri5", "tsit5", "adaptive_heun", "bosh3"):
            return torch.tensor([0.0, 1.0], device=ref.device, dtype=ref.dtype)
        return torch.linspace(0.0, 1.0, self.n_steps + 1, device=ref.device, dtype=ref.dtype)

    def _node(self, key: str, vf: nn.Module) -> nn.Module:
        if key not in self._nodes:
            self._nodes[key] = NeuralODE(
                vf, solver=self.solver, sensitivity=self.sensitivity, atol=self.atol, rtol=self.rtol
            )
        return self._nodes[key]

    def _integrate(self, key: str, vf: nn.Module, x0: Tensor) -> Tensor:
        """Integrate ``vf`` from t=0 to t=1 starting at ``x0``; returns the final state."""
        t_span = self._t_span(x0)
        if _HAS_TORCHDYN:
            _, sol = self._node(key, vf)(x0, t_span)
            return sol[-1]
        if self.solver not in ("rk4", "euler"):
            raise RuntimeError("torchdyn is not installed; only the built-in 'rk4' fallback is available")
        return _rk4(vf, x0, t_span)

    def trajectory(self, x0: Tensor, y=None, eps=None, mask=None, n_eval: int = 11) -> tuple[Tensor, Tensor]:
        """Full forward trajectory (without log-density). Returns (t_eval, states (T, B, d))."""
        self.vf.set_context(y, eps, mask)
        t_span = torch.linspace(0.0, 1.0, n_eval, device=x0.device, dtype=x0.dtype)
        if _HAS_TORCHDYN:
            t_eval, sol = self._node("traj", self.vf)(x0, t_span)
            return t_eval, sol
        xs, x = [x0], x0
        for t0, t1 in zip(t_span[:-1], t_span[1:]):
            x = _rk4(self.vf, x, torch.stack([t0, t1]))
            xs.append(x)
        return t_span, torch.stack(xs)

    # ---- context handling -------------------------------------------------------------
    @staticmethod
    def _expand_context(x: Optional[Tensor], n_samples: int) -> Optional[Tensor]:
        """(B, ...) -> (B * n_samples, ...) by repeating each row ``n_samples`` times."""
        if x is None:
            return None
        return x.repeat_interleave(n_samples, dim=0)

    @staticmethod
    def _log_normal(x: Tensor) -> Tensor:
        return -0.5 * (x**2).sum(-1) - 0.5 * x.shape[-1] * log(2 * pi)

    # ---- public API -------------------------------------------------------------------
    def sample(
        self,
        n_samples: int,
        y: Optional[Tensor] = None,
        eps: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        return_log_prob: bool = False,
        theta0: Optional[Tensor] = None,
    ):
        """Draw posterior samples.

        y    : (B, dy) or (B, n, dy)   — B independent conditioning contexts (None -> prior samples)
        eps  : (B, de) or (B, n, de)
        mask : (B, n)                   — which observations are valid

        Returns theta of shape (B, n_samples, d) when conditioning, else (n_samples, d);
        with ``return_log_prob`` also log q(theta | y, eps) of matching leading shape.
        """
        ref = next(self.prior.parameters())
        conditional = y is not None and self.guidance is not None
        B = y.shape[0] if conditional else 1
        N = B * n_samples
        if theta0 is None:
            theta0 = torch.randn(N, self.dim, device=ref.device, dtype=ref.dtype)
        else:
            theta0 = theta0.reshape(N, self.dim)

        if conditional:
            self.vf.set_context(
                self._expand_context(y, n_samples),
                self._expand_context(eps, n_samples),
                self._expand_context(mask, n_samples),
            )
        else:
            self.vf.set_context(None)

        if return_log_prob:
            self.cnf_fwd.reset_noise(theta0)
            state = torch.cat([theta0, torch.zeros(N, 1, device=ref.device, dtype=ref.dtype)], dim=-1)
            out = self._integrate("fwd", self.cnf_fwd, state)
            theta, dlogp = out[:, :-1], out[:, -1]
            logp = self._log_normal(theta0) + dlogp
            if conditional:
                return theta.reshape(B, n_samples, self.dim), logp.reshape(B, n_samples)
            return theta, logp

        theta = self._integrate("traj", self.vf, theta0)
        return theta.reshape(B, n_samples, self.dim) if conditional else theta

    def log_prob(
        self,
        theta: Tensor,
        y: Optional[Tensor] = None,
        eps: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """log q(theta | y, eps) for theta (B, d) with per-row context y (B, ...), or log p(theta) if y is None.

        Integrates the reversed flow 1 -> 0 and uses  log q(theta) = log N(theta_0) + int_0^1 div v dt.
        """
        theta = theta.reshape(-1, self.dim)
        conditional = y is not None and self.guidance is not None
        if conditional and y.shape[0] != theta.shape[0]:
            raise ValueError("`y` must have one row per theta (expand it with repeat_interleave if needed)")
        self.vf.set_context(y, eps, mask) if conditional else self.vf.set_context(None)

        self.cnf_bwd.reset_noise(theta)
        state = torch.cat([theta, torch.zeros(theta.shape[0], 1, device=theta.device, dtype=theta.dtype)], dim=-1)
        out = self._integrate("bwd", self.cnf_bwd, state)
        theta0, int_div = out[:, :-1], out[:, -1]
        return self._log_normal(theta0) + int_div


__all__ = [
    "exact_divergence",
    "hutchinson_divergence",
    "GuidedVectorField",
    "CNFVectorField",
    "PosteriorSampler",
]
