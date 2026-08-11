#!/usr/bin/env python3
"""
mcmc_regimes.py — Lightweight MCMC for regime-conditional return means.

Why this (not full Bayesian HMM re-estimation):
  The stack already has point-estimate HMM labels + transitions.
  Parameter uncertainty in *within-regime means* is the first-order
  driver of Monte Carlo terminal-wealth dispersion beyond path noise.

Method (Gibbs-style / independent MH per regime):
  For each regime s, asset i:
    likelihood: r_{t in s,i} ~ N(μ, σ²) with σ² fixed at sample variance
    prior: μ ~ N(μ0, τ0²)  (weakly informative around sample mean)
    posterior draws via conjugate Normal-Normal Gibbs

Optional random-walk Metropolis on a row of the transition matrix
(Dirichlet-style proposal on the simplex) to gauge transition uncertainty.

Usage:
  python mcmc_regimes.py --index portfolio --n-draw 2000 --save
  python mcmc_regimes.py --ticker MOS,PFE --n-draw 1000 --save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
OUT = DATA_DIR / "mcmc_regime_means.parquet"
OUT_TRANS = DATA_DIR / "mcmc_transition_draws.parquet"
OUT_SUM = DATA_DIR / "mcmc_regime_summary.parquet"


def _load_rets_and_regimes(tickers: list[str]):
    from monte_carlo import load_returns, load_hmm_alignment, REGIME_ORDER_DEFAULT
    rets = load_returns(tickers).dropna(how="all")
    regimes = load_hmm_alignment(rets.index)
    names = [r for r in REGIME_ORDER_DEFAULT if (regimes == r).any()]
    for r in regimes.dropna().unique():
        if r not in names:
            names.append(r)
    return rets, regimes, names


def gibbs_normal_mean(
    x: np.ndarray,
    n_draw: int,
    burn: int,
    rng: np.random.Generator,
    prior_var_scale: float = 10.0,
) -> np.ndarray:
    """Conjugate Gibbs draws for μ with known variance = sample var."""
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.full(n_draw, np.nanmean(x) if len(x) else 0.0)
    mu_hat = float(x.mean())
    var = float(x.var(ddof=1)) if len(x) > 1 else 1e-6
    var = max(var, 1e-12)
    n = len(x)
    # prior μ ~ N(mu_hat, prior_var_scale * var / n)  weakly centered
    prior_var = prior_var_scale * var / max(n, 1)
    draws = np.zeros(n_draw + burn)
    mu = mu_hat
    for t in range(n_draw + burn):
        # posterior precision
        post_prec = n / var + 1.0 / prior_var
        post_var = 1.0 / post_prec
        post_mean = post_var * (n * mu_hat / var + mu_hat / prior_var)
        mu = rng.normal(post_mean, np.sqrt(post_var))
        draws[t] = mu
    return draws[burn:]


def metropolis_transition_row(
    row: np.ndarray,
    n_draw: int,
    burn: int,
    rng: np.random.Generator,
    step: float = 0.08,
) -> np.ndarray:
    """RW Metropolis on simplex for one transition row (Dirichlet-ish)."""
    row = np.asarray(row, dtype=float)
    row = row / row.sum()
    k = len(row)
    draws = np.zeros((n_draw + burn, k))
    cur = row.copy()
    cur_ll = 0.0  # flat prior on simplex; optional: stickiness prior
    # mild preference for diagonal mass
    def log_prior(p):
        return 2.0 * np.log(p[np.argmax(row)] + 1e-12)  # soft sticky

    cur_lp = log_prior(cur)
    for t in range(n_draw + burn):
        noise = rng.normal(0, step, size=k)
        prop = cur + noise
        prop = np.clip(prop, 1e-6, None)
        prop = prop / prop.sum()
        lp = log_prior(prop)
        if np.log(rng.random()) < (lp - cur_lp):
            cur, cur_lp = prop, lp
        draws[t] = cur
    return draws[burn:]


def run_mcmc(tickers: list[str], n_draw: int = 2000, burn: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rets, regimes, names = _load_rets_and_regimes(tickers)
    cols = list(rets.columns)
    rows = []
    for name in names:
        block = rets.loc[regimes == name]
        for j, col in enumerate(cols):
            x = block[col].values.astype(float)
            draws = gibbs_normal_mean(x, n_draw, burn, rng)
            rows.append({
                "regime": name,
                "ticker": col,
                "n_obs": int(np.sum(~np.isnan(x))),
                "mean_ols": float(np.nanmean(x)),
                "post_mean": float(np.nanmean(draws)),
                "post_std": float(np.nanstd(draws)),
                "p05": float(np.nanquantile(draws, 0.05)),
                "p95": float(np.nanquantile(draws, 0.95)),
            })
    means = pd.DataFrame(rows)

    # transition row MCMC from empirical HMM matrix
    from monte_carlo import load_transition
    P = load_transition(names)
    trans_rows = []
    for i, name in enumerate(names):
        draws = metropolis_transition_row(P[i], n_draw // 2, burn // 2, rng)
        for j, to in enumerate(names):
            trans_rows.append({
                "from_regime": name,
                "to_regime": to,
                "post_mean": float(draws[:, j].mean()),
                "post_std": float(draws[:, j].std()),
                "mle": float(P[i, j]),
            })
    trans = pd.DataFrame(trans_rows)
    return {"means": means, "transitions": trans, "regimes": names}


def main():
    ap = argparse.ArgumentParser(description="MCMC for regime means / transitions")
    try:
        from cli_common import add_index_args, add_ticker_args, add_save_arg, resolve_tickers_from_args
        add_index_args(ap, default="portfolio")
        add_ticker_args(ap)
        add_save_arg(ap)
    except Exception:
        ap.add_argument("--index", action="append")
        ap.add_argument("--ticker")
        ap.add_argument("--save", action="store_true")
    ap.add_argument("--n-draw", type=int, default=2000)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        from cli_common import resolve_tickers_from_args
        tickers = resolve_tickers_from_args(args, default_index="portfolio")
    except Exception:
        tickers = [x.strip().upper() for x in (args.ticker or "MOS,PFE").split(",")]
    print(f"MCMC regimes  tickers={tickers}  draws={args.n_draw}")
    out = run_mcmc(tickers, n_draw=args.n_draw, burn=args.burn, seed=args.seed)
    print(out["means"].to_string(index=False))
    print("\nTransition posterior means:")
    print(out["transitions"].to_string(index=False))
    if args.save:
        out["means"].to_parquet(OUT)
        out["transitions"].to_parquet(OUT_TRANS)
        out["means"].groupby("regime")[["post_mean", "post_std"]].mean().to_parquet(OUT_SUM)
        print(f"Wrote {OUT}\nWrote {OUT_TRANS}\nWrote {OUT_SUM}")


if __name__ == "__main__":
    main()
