#!/usr/bin/env python3
"""
sobol_qmc.py — Sobol' quasi-Monte Carlo sequences for Gaussian shocks.

Sobol points fill [0,1]^d more evenly than i.i.d. uniform (low discrepancy).
Mapped through Φ^{-1} they become quasi-Gaussian samples for MC integration.

Implementation notes
--------------------
* Prefer SciPy's `qmc.Sobol` with **Owen scrambling** (scramble=True):
  - destroys lattice artifacts
  - allows independent randomizations (seeds) for error estimates
* Sample counts that are powers of 2 match Sobol's dyadic construction
  (`random_base2`). We draw 2^ceil(log2(n)) and truncate.
* Dimension d = n_assets * horizon can be large; Sobol supports high d but
  quality degrades for very high d — prefer path-wise Brownian bridging
  or factor structure if d >> 1000 (future work).
* Fallback: scrambled Halton or plain Normal if SciPy missing.

Usage:
  from sobol_qmc import sobol_normal
  Z = sobol_normal(n_paths=1024, dim=8, seed=0)  # shape (1024, 8)
"""
from __future__ import annotations

import numpy as np


def _next_pow2(n: int) -> int:
    n = max(int(n), 1)
    return 1 << int(np.ceil(np.log2(n)))


def sobol_uniform(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Owen-scrambled Sobol uniforms in (0,1)^dim, shape (n, dim)."""
    dim = int(dim)
    n = int(n)
    try:
        from scipy.stats import qmc
        m = int(np.ceil(np.log2(max(n, 2))))
        engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
        u = engine.random_base2(m)  # 2^m rows
        u = u[:n]
        return np.clip(u, 1e-12, 1 - 1e-12)
    except Exception:
        # Fallback: stratified Latin-like jittered grid + shuffle per dim
        rng = np.random.default_rng(seed)
        u = np.zeros((n, dim))
        for j in range(dim):
            edges = np.linspace(0, 1, n + 1)
            u[:, j] = rng.uniform(edges[:-1], edges[1:])
            rng.shuffle(u[:, j])
        return np.clip(u, 1e-12, 1 - 1e-12)


def sobol_normal(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Gaussian quasi-samples via Sobol + inverse CDF, shape (n, dim)."""
    u = sobol_uniform(n, dim, seed=seed)
    try:
        from scipy.stats import norm
        return norm.ppf(u)
    except Exception:
        # rational approximation (Abramowitz & Stegun 26.2.23 style rough)
        # use erfinv if available
        return np.sqrt(2) * _erfinv_approx(2 * u - 1)


def _erfinv_approx(y: np.ndarray) -> np.ndarray:
    """Crude erfinv for fallback only."""
    a = 0.147  # Winitzki approximation
    sgn = np.sign(y)
    ln = np.log(1 - y * y)
    t = 2 / (np.pi * a) + ln / 2
    return sgn * np.sqrt(np.sqrt(t * t - ln / a) - t)


def sobol_normal_tensor(
    n_paths: int, horizon: int, n_assets: int, seed: int = 0
) -> np.ndarray:
    """Shape (n_paths, horizon, n_assets) quasi-Gaussian shocks."""
    Z = sobol_normal(n_paths * horizon, n_assets, seed=seed)
    return Z.reshape(n_paths, horizon, n_assets)


def discrepancy_report(n: int = 512, dim: int = 4, seed: int = 0) -> dict:
    """Compare crude discrepancy proxy: mean pairwise spacing variance."""
    rng = np.random.default_rng(seed)
    u_mc = rng.random((n, dim))
    u_q = sobol_uniform(n, dim, seed=seed)
    def spread(u):
        # variance of coordinate-wise means from 0.5 (should be smaller for QMC)
        return float(np.mean((u.mean(axis=0) - 0.5) ** 2))
    return {
        "mc_mean_sq_dev": spread(u_mc),
        "sobol_mean_sq_dev": spread(u_q),
        "n": n,
        "dim": dim,
    }


if __name__ == "__main__":
    r = discrepancy_report()
    print("Sobol vs MC mean-square deviation from 0.5:", r)
    Z = sobol_normal(256, 3, seed=1)
    print("Z shape", Z.shape, "mean", Z.mean(axis=0), "std", Z.std(axis=0))
