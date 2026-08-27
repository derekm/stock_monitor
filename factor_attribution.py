#!/usr/bin/env python3
"""
factor_attribution.py — Daily factor attribution of portfolio/index returns.

Uses FF5+MOM factors from factor_library.py to decompose returns into:
- MKT (market beta)
- SMB (size)
- HML (value)
- RMW (profitability)
- CMA (investment)
- MOM (momentum)
- Alpha (residual)

For any portfolio weights or index level series.
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
    prices = pd.read_parquet(DATA_DIR / "daily_prices/")
    if prices["date"].dtype != "datetime64[ns]":
        prices["date"] = pd.to_datetime(prices["date"])
    close = prices.pivot(index="date", columns="ticker", values="close")
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    rets = close.pct_change()
    return rets


def load_portfolio_weights() -> pd.DataFrame | None:
    """Load portfolio weights if available (e.g., bogle_tmi, shadow_book)."""
    # Try various portfolio files
    for fname in ["bogle_tmi.parquet", "bogle_qmi.parquet", "bogle_bpi.parquet", 
                  "shadow_book.parquet", "portfolio_holdings.parquet"]:
        path = DATA_DIR / fname
        if path.exists():
            df = pd.read_parquet(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            # If it has 'level' column, compute returns from level
            if "level" in df.columns:
                rets = df["level"].pct_change().dropna()
                return rets.to_frame("portfolio_ret")
            return df
    return None


def compute_portfolio_returns(
    weights_or_rets: pd.DataFrame,
    rets: pd.DataFrame
) -> pd.Series:
    """Compute portfolio returns from weights and asset returns, or use provided returns."""
    # If input already has portfolio_ret column, use it
    if "portfolio_ret" in weights_or_rets.columns:
        return weights_or_rets["portfolio_ret"]
    
    # Otherwise treat as weights
    weights = weights_or_rets
    # Align
    common_dates = weights.index.intersection(rets.index)
    common_tickers = weights.columns.intersection(rets.columns)
    
    if len(common_dates) == 0 or len(common_tickers) == 0:
        return pd.Series(dtype=float)
    
    w = weights.loc[common_dates, common_tickers]
    r = rets.loc[common_dates, common_tickers]
    
    # Forward fill weights, normalize to 1
    w = w.ffill().fillna(0)
    w = w.div(w.sum(axis=1), axis=0).fillna(0)
    
    port_rets = (w * r).sum(axis=1)
    return port_rets


def run_factor_regression(
    port_rets: pd.Series,
    factors: pd.DataFrame,
    window: int = 252
) -> pd.DataFrame:
    """Rolling factor regression."""
    common = port_rets.index.intersection(factors.index)
    port_rets = port_rets.loc[common]
    factors = factors.loc[common]
    
    results = []
    
    for i in range(window, len(common)):
        y = port_rets.iloc[i-window:i].values
        X = factors.iloc[i-window:i].values
        
        mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        if mask.sum() < 60:
            continue
            
        try:
            names = list(factors.columns)
            reg = LinearRegression().fit(X[mask], y[mask])
            alpha = reg.intercept_ * 252
            betas = pd.Series(reg.coef_, index=names)
            r2 = reg.score(X[mask], y[mask])
            today = factors.iloc[i].reindex(names)
            contrib = betas * today
            row = {"date": common[i], "alpha_ann": alpha, "r2": r2}
            for n in names:
                row[f"beta_{n}"] = float(betas[n])
                row[f"factor_contrib_{n}"] = float(contrib[n]) * 252
            row["residual_ann"] = float(y[mask][-1] - (reg.intercept_ + contrib.sum())) * 252
            results.append(row)
        except:
            continue
    
    return pd.DataFrame(results).set_index("date")


def main():
    ap = argparse.ArgumentParser(description="Factor attribution for portfolio/index")
    ap.add_argument("--portfolio", choices=["bogle_tmi", "bogle_qmi", "bogle_bpi", "auto"],
                    default="auto", help="Portfolio to attribute")
    ap.add_argument("--save", action="store_true", help="Save outputs")
    ap.add_argument("--window", type=int, default=252, help="Rolling window (days)")
    args = ap.parse_args()

    print("Loading factors...")
    factors = load_factors()
    print(f"  Factors: {factors.columns.tolist()}, {len(factors)} dates")

    print("Loading returns...")
    rets = load_returns()
    print(f"  Returns: {rets.shape[1]} tickers, {len(rets)} dates")

    print("Loading portfolio...")
    weights = load_portfolio_weights()
    if weights is None:
        print("  No portfolio weights found, using equal-weight universe")
        # Create equal-weight universe
        weights = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
        weights = weights.div(weights.sum(axis=1), axis=0)

    print(f"  Weights: {weights.shape}")

    print("Computing portfolio returns...")
    port_rets = compute_portfolio_returns(weights, rets)
    print(f"  Portfolio returns: {len(port_rets)} dates")

    print("Running factor regression...")
    attrib = run_factor_regression(port_rets, factors, args.window)
    print(f"  Attribution: {len(attrib)} dates")

    if args.save:
        out_path = DATA_DIR / f"factor_attribution_{args.portfolio}.parquet"
        attrib.to_parquet(out_path)
        print(f"\nSaved {out_path}")

    print("\n=== Latest Attribution ===")
    if len(attrib) > 0:
        print(attrib.tail(1).T.to_string())

    return attrib


if __name__ == "__main__":
    main()