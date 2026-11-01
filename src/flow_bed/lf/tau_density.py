"""Piecewise-linear flow-time densities rho(tau) on a grid (e.g. rho ∝ design-gradient RMS from analysis/tau_rms.py)."""

from __future__ import annotations

import json

import numpy as np


def rho_nodes(taus: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """Values of the piecewise-linear density rho ∝ rms at the grid nodes (∫ rho = 1, trapezoid = exact)."""
    return rms / np.trapezoid(rms, taus)


def sample_tau(taus: np.ndarray, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse CDF of the piecewise-linear density with node values rho on taus, for uniforms u."""
    dt = np.diff(taus)
    seg = dt * (rho[:-1] + rho[1:]) / 2
    cdf = np.concatenate([[0], np.cumsum(seg)])
    k = np.clip(np.searchsorted(cdf, u, side="right") - 1, 0, len(dt) - 1)
    r = u - cdf[k]  # mass needed inside segment k: a s + b s^2 / 2 = r, s = tau - tau_k
    a, b = rho[k], (rho[k + 1] - rho[k]) / dt[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(np.abs(b) > 1e-12, (np.sqrt(np.maximum(a**2 + 2 * b * r, 0)) - a) / b, r / np.maximum(a, 1e-300))
    return taus[k] + np.clip(s, 0, dt[k])


def load_rho(path: str, key: str = "policy_mix", div: str = "hu") -> tuple[np.ndarray, np.ndarray]:
    """(taus, rho at the nodes) from a JSON of analysis/tau_rms.py, by default policy designs with q ~ U{1..30} and
    the Hutchinson divergence."""
    with open(path) as f:
        d = json.load(f)
    taus = np.array(d["taus"])
    return taus, rho_nodes(taus, np.array(d["results"][key][div]["rms"]))
