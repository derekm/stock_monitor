#!/usr/bin/env python3
"""
robust_covariance.py — Robust covariance estimators for portfolio optimization.

Estimators:
  sample     — classical sample covariance
  ledoit_wolf — Ledoit-Wolf linear shrinkage toward constant-correlation or identity
  oas        — Oracle Approximating Shrinkage
  ewma       — Exponentially weighted cov (RiskMetrics-style)
  huber      — Simple robust cov via winsorized returns

Usage:
  python robust_covariance.py --universe portfolio
  python robust_covariance.py --universe growth --window 126 --save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cli_common import (
    add_index_args, add_ticker_args, add_sector_arg, add_save_arg,
    add_window_arg, resolve_tickers_from_args, resolve_index_names_from_args,
    build_parser,
)
from index_registry import parse_indexes, tickers_for_index, available_indexes, index_help_text

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
HOLDINGS = DATA_DIR / "portfolio_holdings.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT = DATA_DIR / "robust_covariance_summary.parquet"


def load_returns(tickers: list[str], window: int = 126) -> pd.DataFrame:
    prices = pd.read_parquet(PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    wide = (
        prices[prices["ticker"].isin(tickers)]
        .pivot_table(index="date", columns="ticker", values="close")
        .sort_index()
        .ffill()
    )
    rets = np.log(wide / wide.shift(1)).dropna(how="all").iloc[-window:]
    return rets.dropna(axis=1, thresh=max(40, window // 3))


def sample_cov(rets: pd.DataFrame) -> np.ndarray:
    return rets.cov().values * 252.0


def ledoit_wolf_cov(rets: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage toward scaled identity (analytical)."""
    X = rets.values
    T, N = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    S = (X.T @ X) / T  # sample cov (daily)
    mu = np.trace(S) / N
    F = mu * np.eye(N)
    # Frobenius
    d2 = np.sum((S - F) ** 2)
    # estimator of pi, rho simplified
    X2 = X ** 2
    pi_mat = (X2.T @ X2) / T - S ** 2
    pi_hat = np.sum(pi_mat)
    # rho for identity target
    rho_hat = np.sum(np.diag(pi_mat))
    kappa = (pi_hat - rho_hat) / d2 if d2 > 0 else 0.0
    shrink = float(np.clip(kappa / T, 0.0, 1.0))
    Sigma = shrink * F + (1 - shrink) * S
    return Sigma * 252.0, shrink


def oas_cov(rets: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Oracle Approximating Shrinkage (Chen et al.)."""
    X = rets.values
    T, N = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    S = (X.T @ X) / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)
    fro = np.sum((S - F) ** 2)
    num = np.sum(S ** 2) + (np.trace(S)) ** 2
    denom = (T + 1.0) * fro if fro > 0 else 1.0
    # OAS formula (simplified)
    shrink = float(np.clip((num / N) / denom, 0.0, 1.0))
    # more standard OAS:
    rho = min(1.0, (1.0 - 2.0 / N) * fro / ((T + 1.0) * fro) if fro else 1.0)
    # Chen OAS:
    alpha = (1 - 2 / N) * fro + (np.trace(S)) ** 2
    shrink = float(np.clip(alpha / ((T + 1 - 2 / N) * fro) if fro else 1.0, 0.0, 1.0))
    Sigma = shrink * F + (1 - shrink) * S
    return Sigma * 252.0, shrink


def ewma_cov(rets: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    X = rets.values
    T, N = X.shape
    S = np.cov(X[: max(20, N + 5)].T)
    for t in range(max(20, N + 5), T):
        r = X[t : t + 1].T
        S = lam * S + (1 - lam) * (r @ r.T)
    return S * 252.0


def winsorized_cov(rets: pd.DataFrame, z: float = 2.5) -> np.ndarray:
    from analytics_common import winsor_z
    return winsor_z(rets, z).cov().values * 252.0


def condition_number(S: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(S)
    ev = ev[ev > 1e-12]
    return float(ev.max() / ev.min()) if len(ev) else float("nan")


def resolve(universe: str) -> list[str]:
    try:
        names = parse_indexes(universe)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    seen, out = set(), []
    for n in names:
        for tk in tickers_for_index(n):
            if tk not in seen:
                seen.add(tk)
                out.append(tk)
    if not out:
        raise SystemExit(f"No tickers for {universe!r}. Available: {available_indexes()}")
    return out



def run(universe: str = "portfolio", window: int = 126, save: bool = False) -> dict:
    tickers = resolve(universe)
    rets = load_returns(tickers, window)
    tickers = list(rets.columns)
    print(f"Robust covariance · {universe} · n={len(tickers)} · window={window}")

    estimators = {}
    S_sample = sample_cov(rets)
    estimators["sample"] = (S_sample, 0.0)
    S_lw, sh_lw = ledoit_wolf_cov(rets)
    estimators["ledoit_wolf"] = (S_lw, sh_lw)
    S_oas, sh_oas = oas_cov(rets)
    estimators["oas"] = (S_oas, sh_oas)
    estimators["ewma"] = (ewma_cov(rets), 0.0)
    estimators["winsorized"] = (winsorized_cov(rets), 0.0)

    rows = []
    for name, (S, shrink) in estimators.items():
        vols = np.sqrt(np.maximum(np.diag(S), 0))
        rows.append({
            "universe": universe,
            "estimator": name,
            "shrinkage": shrink,
            "cond_number": condition_number(S),
            "avg_vol": float(np.mean(vols)),
            "median_vol": float(np.median(vols)),
            "min_eig": float(np.linalg.eigvalsh(S).min()),
            "frobenius_vs_sample": float(np.linalg.norm(S - S_sample, "fro")),
        })
        print(f"  {name:12s} shrink={shrink:.3f}  cond={condition_number(S):8.1f}  "
              f"avg_vol={np.mean(vols)*100:.1f}%  min_eig={np.linalg.eigvalsh(S).min():.4f}")

    df = pd.DataFrame(rows)
    if save:
        df.to_parquet(OUT)
        # save LW cov matrix
        pd.DataFrame(S_lw, index=tickers, columns=tickers).to_parquet(DATA_DIR / f"cov_ledoit_wolf_{universe}.parquet")
        print(f"Wrote {OUT}")
    return {"tickers": tickers, "estimators": estimators, "summary": df}


def main():
    ap = argparse.ArgumentParser()
    add_index_args(ap, default="portfolio")
    ap.add_argument("--window", type=int, default=126)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run((','.join(resolve_index_names_from_args(args, default_index='portfolio')) or 'portfolio'), args.window, args.save)


if __name__ == "__main__":
    main()
