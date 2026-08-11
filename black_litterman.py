#!/usr/bin/env python3
"""
black_litterman.py — Black-Litterman expected returns & posterior weights.

Steps:
  1. Equilibrium returns  π = δ Σ w_mkt  (reverse optimization)
  2. Investor views      P μ = Q + ε,  ε ~ N(0, Ω)
  3. Posterior           μ_bl = [(τΣ)^{-1} + P' Ω^{-1} P]^{-1} [(τΣ)^{-1} π + P' Ω^{-1} Q]
  4. Mean-variance weights with μ_bl (long-only optional)

Usage:
  python black_litterman.py --universe portfolio
  python black_litterman.py --universe portfolio --view NVDA:0.05 --view PFE:0.08
  python black_litterman.py --universe growth --tau 0.05 --delta 2.5 --save

Views format: TICKER:excess_return  (absolute view on single asset)
"""

from __future__ import annotations

import argparse
from cli_common import (
    add_index_args, add_ticker_args, add_sector_arg, add_save_arg,
    add_window_arg, resolve_tickers_from_args, resolve_index_names_from_args,
    build_parser,
)
from pathlib import Path

import numpy as np
import pandas as pd

from robust_covariance import load_returns, ledoit_wolf_cov, resolve, sample_cov

DATA_DIR = Path(__file__).parent
OUT = DATA_DIR / "black_litterman_weights.parquet"

from scipy.optimize import minimize  # noqa: F401  (canonical availability flag in analytics_common)
from analytics_common import HAS_SCIPY  # canonical scipy-availability flag


def black_litterman(
    Sigma: np.ndarray,
    w_eq: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray | None = None,
    tau: float = 0.05,
    delta: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (mu_bl, posterior_cov_of_mu).
    Sigma: annualized cov of returns
    w_eq: equilibrium market weights
    P: k x n pick matrix
    Q: k views
    """
    n = Sigma.shape[0]
    pi = delta * Sigma @ w_eq  # equilibrium excess returns
    tau_S = tau * Sigma
    if omega is None:
        # He-Litterman: proportional to uncertainty of view
        omega = np.diag(np.diag(P @ tau_S @ P.T))
        omega = np.maximum(omega, 1e-12)

    tau_S_inv = np.linalg.pinv(tau_S)
    omega_inv = np.linalg.pinv(omega)
    M = tau_S_inv + P.T @ omega_inv @ P
    mu_bl = np.linalg.pinv(M) @ (tau_S_inv @ pi + P.T @ omega_inv @ Q)
    return mu_bl, pi


def mv_long_only(mu: np.ndarray, Sigma: np.ndarray, risk_aversion: float = 2.5) -> np.ndarray:
    n = len(mu)
    if HAS_SCIPY:
        def obj(w):
            return -float(w @ mu - 0.5 * risk_aversion * w @ Sigma @ w)
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        res = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-14})
        w = np.maximum(res.x, 0)
        return w / w.sum()
    # unconstrained then project
    try:
        w = np.linalg.pinv(risk_aversion * Sigma) @ mu
    except Exception:
        w = np.ones(n) / n
    w = np.maximum(w, 0)
    return w / w.sum() if w.sum() > 0 else np.ones(n) / n


def parse_views(view_args: list[str], tickers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    qs = []
    for v in view_args:
        if ":" not in v:
            continue
        t, q = v.split(":", 1)
        t = t.strip().upper()
        if t not in tickers:
            print(f"  skip view {t}: not in universe")
            continue
        rows.append(t)
        qs.append(float(q))
    k, n = len(rows), len(tickers)
    P = np.zeros((k, n))
    for i, t in enumerate(rows):
        P[i, tickers.index(t)] = 1.0
    return P, np.array(qs)


def run(universe: str, views: list[str], tau: float, delta: float, window: int, save: bool):
    tickers = resolve(universe)
    rets = load_returns(tickers, window)
    tickers = list(rets.columns)
    Sigma, shrink = ledoit_wolf_cov(rets)
    # eq weights: if portfolio, use holdings; else EW
    if universe == "portfolio" and (DATA_DIR / "portfolio_holdings.parquet").exists():
        h = pd.read_parquet(DATA_DIR / "portfolio_holdings.parquet")
        cw = h.set_index("ticker")["weight"].astype(float)
        if cw.sum() > 2:
            cw = cw / 100.0
        w_eq = np.array([float(cw.get(t, 0)) for t in tickers])
        if w_eq.sum() <= 0:
            w_eq = np.ones(len(tickers)) / len(tickers)
        else:
            w_eq = w_eq / w_eq.sum()
    else:
        w_eq = np.ones(len(tickers)) / len(tickers)

    pi = delta * Sigma @ w_eq
    if views:
        P, Q = parse_views(views, tickers)
        if len(Q) == 0:
            print("No valid views — using equilibrium only")
            mu_bl = pi
        else:
            mu_bl, _ = black_litterman(Sigma, w_eq, P, Q, tau=tau, delta=delta)
            print(f"Views applied: {list(zip([tickers[j] for j in P.argmax(1)], Q))}")
    else:
        # default illustrative views: mild overweight selected names
        defaults = []
        if "PFE" in tickers:
            defaults.append("PFE:0.08")
        if "MOS" in tickers:
            defaults.append("MOS:0.10")
        if defaults:
            print(f"No --view given; using defaults {defaults}")
            P, Q = parse_views(defaults, tickers)
            mu_bl, _ = black_litterman(Sigma, w_eq, P, Q, tau=tau, delta=delta)
        else:
            mu_bl = pi

    w_bl = mv_long_only(mu_bl, Sigma, risk_aversion=delta)
    w_eq_mv = mv_long_only(pi, Sigma, risk_aversion=delta)

    print(f"\nBlack-Litterman · {universe} · τ={tau} · δ={delta} · LW shrink used")
    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "ticker": t,
            "w_eq": float(w_eq[i]),
            "pi": float(pi[i]),
            "mu_bl": float(mu_bl[i]),
            "w_bl": float(w_bl[i]),
            "w_eq_mv": float(w_eq_mv[i]),
            "delta_w": float(w_bl[i] - w_eq[i]),
        })
    df = pd.DataFrame(rows)
    print(df.assign(
        w_eq=lambda d: (d.w_eq * 100).round(2),
        w_bl=lambda d: (d.w_bl * 100).round(2),
        w_eq_mv=lambda d: (d.w_eq_mv * 100).round(2),
        pi=lambda d: (d.pi * 100).round(2),
        mu_bl=lambda d: (d.mu_bl * 100).round(2),
        delta_w=lambda d: (d.delta_w * 100).round(2),
    ).to_string(index=False))

    if save:
        df["universe"] = universe
        df.to_parquet(OUT)
        print(f"Wrote {OUT}")
    return df


def main():
    ap = argparse.ArgumentParser()
    add_index_args(ap, default="portfolio")
    ap.add_argument("--view", action="append", default=[], help="TICKER:return e.g. NVDA:-0.02")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=2.5)
    ap.add_argument("--window", type=int, default=126)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run((','.join(resolve_index_names_from_args(args, default_index='portfolio')) or 'portfolio'), args.view, args.tau, args.delta, args.window, args.save)


if __name__ == "__main__":
    main()
