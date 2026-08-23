#!/usr/bin/env python3
"""
factor_adjust_signals.py — Factor-adjust signal family scores from signal_aggregator.

Loads signal_aggregator_scores.parquet (family scores per ticker), regresses each
family score on FF5+MOM factors (using ticker returns), computes residual scores.

Outputs:
- signal_factor_loadings.parquet: beta per family per factor
- signal_residual_scores.parquet: factor-adjusted family scores
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_DIR = Path(__file__).parent


def load_factors() -> pd.DataFrame:
    """Load FF5+MOM factors."""
    factors = pd.read_parquet(DATA_DIR / "ff5_factors.parquet")
    factors.index = pd.to_datetime(factors.index)
    return factors


def load_returns() -> pd.DataFrame:
    """Load daily returns (date × ticker)."""
    prices = pd.read_parquet(DATA_DIR / "daily_prices.parquet")
    if prices["date"].dtype != "datetime64[ns]":
        prices["date"] = pd.to_datetime(prices["date"])
    close = prices.pivot(index="date", columns="ticker", values="close")
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    rets = close.pct_change().shift(-1)  # next-day returns
    return rets


def load_signal_scores() -> pd.DataFrame:
    """Load signal aggregator family scores."""
    scores = pd.read_parquet(DATA_DIR / "signal_aggregator_scores.parquet")
    return scores


def factor_adjust_family_scores(
    scores: pd.DataFrame,
    factors: pd.DataFrame,
    rets: pd.DataFrame,
    family_cols: list[str],
    lookback: int = 252,
    min_obs: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each ticker and family score:
    1. Get ticker's return history
    2. Regress family score on factors (using factor returns as X, ticker return as proxy)
    3. Actually: we want to regress the FAMILY SIGNAL (cross-sectional) on FACTORS
    
    Better approach: For each date, we have cross-sectional family scores across tickers.
    Regress cross-section of family scores on cross-section of factor exposures.
    """
    
    # Align dates
    common_dates = scores.index.intersection(factors.index) if hasattr(scores, 'index') else []
    
    # Actually, signal_aggregator_scores is cross-sectional (one row per ticker, not per date)
    # So we need a different approach: regress each family's cross-sectional scores
    # on factor loadings (betas) computed from historical returns.
    
    # Step 1: Compute factor betas for each ticker (rolling)
    tickers = scores["ticker"].unique()
    factor_betas = {}
    
    common_rets_dates = rets.index.intersection(factors.index)
    rets_aligned = rets.loc[common_rets_dates]
    factors_aligned = factors.loc[common_rets_dates]
    
    print(f"Computing factor betas for {len(tickers)} tickers...")
    
    for ticker in tickers:
        if ticker not in rets_aligned.columns:
            continue
        ticker_rets = rets_aligned[ticker].dropna()
        if len(ticker_rets) < min_obs:
            continue
        
        # Align with factors
        common = ticker_rets.index.intersection(factors_aligned.index)
        if len(common) < min_obs:
            continue
            
        y = ticker_rets.loc[common].values
        X = factors_aligned.loc[common].values
        
        # Full sample regression (or rolling - use last window)
        window = min(lookback, len(common))
        y_win = y[-window:]
        X_win = X[-window:]
        
        mask = ~np.isnan(y_win) & ~np.any(np.isnan(X_win), axis=1)
        if mask.sum() < min_obs:
            continue
            
        try:
            reg = LinearRegression().fit(X_win[mask], y_win[mask])
            factor_betas[ticker] = reg.coef_
        except:
            continue
    
    betas_df = pd.DataFrame.from_dict(factor_betas, orient="index", columns=[f"beta_{f}" for f in factors.columns])
    betas_df.index.name = "ticker"
    
    # Step 2: Regress each family score on factor betas (cross-sectional)
    loadings_list = []
    # Initialize residual columns with NaN, aligned to scores index
    residual_scores = pd.DataFrame(index=scores.index)
    residual_scores["ticker"] = scores["ticker"]
    
    for family in family_cols:
        if family not in scores.columns:
            continue
            
        # Merge scores with betas (LEFT join to preserve all scores)
        merged = scores[["ticker", family]].merge(betas_df, left_on="ticker", right_index=True, how="left")
        if len(merged) < 20:
            continue
            
        # Only use rows with valid betas
        beta_cols = [c for c in merged.columns if c.startswith("beta_")]
        valid_mask = merged[beta_cols].notna().all(axis=1) & merged[family].notna()
        if valid_mask.sum() < 20:
            continue
            
        y = merged.loc[valid_mask, family].values
        X = merged.loc[valid_mask, beta_cols].values
        
        try:
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            residuals = y - y_pred
            
            # Store loadings
            loadings_list.append({
                "family": family,
                **{f"beta_{f}": b for f, b in zip(factors.columns, reg.coef_)},
                "r2": reg.score(X, y)
            })
            
            # Store residuals (aligned to scores index via merged ticker order)
            residual_scores[f"{family}_residual"] = np.nan
            residual_scores.loc[merged.index[valid_mask], f"{family}_residual"] = residuals
            
            print(f"  {family}: R²={reg.score(X, y):.3f}, n={len(y)}, betas={dict(zip(factors.columns, reg.coef_))}")
        except Exception as e:
            print(f"  {family}: failed - {e}")
    
    loadings_df = pd.DataFrame(loadings_list)
    return loadings_df, residual_scores


def main():
    ap = argparse.ArgumentParser(description="Factor-adjust signal family scores")
    ap.add_argument("--families", nargs="+", default=["preferred", "peer", "cross", "earnings"],
                    help="Signal families to adjust")
    ap.add_argument("--save", action="store_true", help="Save outputs")
    args = ap.parse_args()

    print("Loading factors...")
    factors = load_factors()
    print(f"  Factors: {factors.columns.tolist()}, {len(factors)} dates")

    print("Loading returns...")
    rets = load_returns()
    print(f"  Returns: {rets.shape[1]} tickers, {len(rets)} dates")

    print("Loading signal scores...")
    scores = load_signal_scores()
    print(f"  Scores: {len(scores)} tickers, families: {[c for c in scores.columns if c not in ['ticker', 'sector', 'in_portfolio', 'life_cycle_stage', 'rank', 'composite']]}")

    print("\nFactor-adjusting family scores...")
    loadings, residuals = factor_adjust_family_scores(scores, factors, rets, args.families)

    if args.save:
        loadings_path = DATA_DIR / "signal_factor_loadings.parquet"
        residuals_path = DATA_DIR / "signal_residual_scores.parquet"
        loadings.to_parquet(loadings_path)
        residuals.to_parquet(residuals_path)
        print(f"\nSaved {loadings_path}")
        print(f"Saved {residuals_path}")

    print("\n=== Factor Loadings ===")
    print(loadings.to_string(index=False))

    return loadings, residuals


if __name__ == "__main__":
    main()