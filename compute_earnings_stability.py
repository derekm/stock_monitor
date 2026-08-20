#!/usr/bin/env python3
"""
compute_earnings_stability.py — Compute earnings_stability for all tickers.

Earnings stability = R-squared of linear trend through quarterly net income values.
Higher = more predictable, stable earnings growth.
Lower = erratic or unpredictable earnings.

Method:
1. For each ticker, get time series of quarterly net income
2. Fit linear trend through available data (need >= 4 quarters)
3. Compute R-squared of actual vs trend line
4. Store as earnings_stability (0-1 range, higher = more stable)

Usage:
  python compute_earnings_stability.py
  python compute_earnings_stability.py --min-quarters 8
"""

import pandas as pd
from analytics_common import atomic_write_parquet
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"


def compute_earnings_stability_for_ticker(group: pd.DataFrame, min_quarters: int = 4) -> pd.Series:
    """
    Compute earnings stability for a single ticker's time series.
    
    Returns Series with earnings_stability value for each row in the group.
    """
    group = group.sort_values("as_of_date")
    
    # Get net income series
    ni_values = group["net_income_quarterly"].dropna()
    
    if len(ni_values) < min_quarters:
        # Not enough data for trend analysis
        return pd.Series(np.nan, index=group.index)
    
    # Fit linear trend
    x = np.arange(len(ni_values))
    y = ni_values.values
    
    # Handle case where all values are zero or same
    if np.std(y) < 1e-10:
        return pd.Series(1.0, index=group.index)
    
    # Linear regression
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    
    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    if ss_tot < 1e-10:
        r_squared = 1.0
    else:
        r_squared = 1 - (ss_res / ss_tot)
    
    # Clamp to [0, 1]
    r_squared = np.clip(r_squared, 0, 1)
    
    return pd.Series(r_squared, index=group.index)


def main(min_quarters: int = 4):
    print("Loading fundamentals...")
    fund = pd.read_parquet(FUND)
    fund["as_of_date"] = pd.to_datetime(fund["as_of_date"]).dt.date
    
    print(f"Total rows: {len(fund)}")
    print(f"Tickers: {fund['ticker'].nunique()}")
    
    # Compute earnings stability for each ticker
    print(f"\nComputing earnings stability (min {min_quarters} quarters)...")
    
    results = []
    for ticker, group in fund.groupby("ticker"):
        stability = compute_earnings_stability_for_ticker(group, min_quarters)
        group = group.copy()
        group["earnings_stability"] = stability
        group["earnings_stability_provenance"] = stability.apply(
            lambda x: "computed" if pd.notna(x) else "insufficient_data"
        )
        results.append(group)
    
    fund_out = pd.concat(results, ignore_index=True)
    
    # Stats
    has_stability = fund_out["earnings_stability"].notna().sum()
    print(f"\nEarnings stability computed: {has_stability}/{len(fund_out)} rows ({has_stability/len(fund_out)*100:.1f}%)")
    print(f"  Mean: {fund_out['earnings_stability'].mean():.3f}")
    print(f"  Median: {fund_out['earnings_stability'].median():.3f}")
    
    # Save
    atomic_write_parquet(fund_out, FUND)
    print(f"\nSaved to {FUND}")
    
    return fund_out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-quarters", type=int, default=4)
    args = ap.parse_args()
    
    main(min_quarters=args.min_quarters)
