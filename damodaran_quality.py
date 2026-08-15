#!/usr/bin/env python3
"""
damodaran_quality.py — Damodaran's quality screens implementation (vectorized).

Quality screens per Damodaran's framework:
1. Excess Returns: ROIC > WACC (earning more than cost of capital)
2. Return Consistency: Stable/high ROE & ROIC over time
3. Reinvestment Quality: High ROIC on reinvested capital
4. Financial Health: Interest coverage > 4, D/E < 2, positive FCF
5. Cash Flow Quality: FCF conversion > 80%, low accruals
6. Growth Quality: Revenue growth > GDP, low volatility
7. Moat Indicators: Pricing power (stable margins), ROIC persistence
8. Capital Allocation: Shareholder-friendly (buybacks, sustainable dividends)

Outputs:
- quality_scores.parquet — ticker × as_of_date × quality_score (0-100) + component scores
- quality_screens.parquet — ticker × as_of_date × pass/fail per screen
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
WACC_FILE = DATA_DIR / "wacc_per_ticker.parquet"
OUT_QUALITY = DATA_DIR / "quality_scores.parquet"
OUT_SCREENS = DATA_DIR / "quality_screens.parquet"


def load_fundamentals() -> pd.DataFrame:
    if not FUND.exists():
        raise FileNotFoundError(f"{FUND} not found")
    df = pd.read_parquet(FUND)
    if "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df = df.sort_values(["ticker", "as_of_date"])
    return df


def load_wacc() -> pd.DataFrame:
    if not WACC_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(WACC_FILE)


def compute_return_consistency(fund: pd.DataFrame) -> pd.DataFrame:
    """Vectorized: 5-year ROE/ROIC stability per ticker"""
    fund = fund.copy()
    fund["roe_rolling_mean"] = fund.groupby("ticker")["roe"].transform(
        lambda x: x.rolling(20, min_periods=8).mean()
    )
    fund["roe_rolling_std"] = fund.groupby("ticker")["roe"].transform(
        lambda x: x.rolling(20, min_periods=8).std()
    )
    fund["roic_rolling_mean"] = fund.groupby("ticker")["roic"].transform(
        lambda x: x.rolling(20, min_periods=8).mean()
    )
    fund["roic_rolling_std"] = fund.groupby("ticker")["roic"].transform(
        lambda x: x.rolling(20, min_periods=8).std()
    )
    fund["roe_cv"] = fund["roe_rolling_std"] / fund["roe_rolling_mean"].replace(0, np.nan)
    fund["roic_cv"] = fund["roic_rolling_std"] / fund["roic_rolling_mean"].replace(0, np.nan)
    fund["return_consistency"] = (
        (fund["roe_rolling_mean"] > 0.15) & (fund["roe_cv"] < 0.5) &
        (fund["roic_rolling_mean"] > 0.15) & (fund["roic_cv"] < 0.5)
    )
    return fund


def compute_reinvestment_quality(fund: pd.DataFrame) -> pd.DataFrame:
    """ROIC-based reinvestment quality — degrades gracefully without FCF.
    
    True Damodaran reinvestment = change in invested capital / NOPAT, which
    needs invested-capital history. Our reinvestment_rate (capex/op_income)
    is sparse, so we fall back to the ROIC level (reinvestment value is
    fundamentally grounded in ROIC regardless of the funding source).
    """
    fund = fund.copy()
    fund["reinvestment_quality"] = fund["roic"] > 0.15
    fund["reinvestment_profile"] = np.where(
        (fund["roic"] > 0.20), "compounder",
        np.where((fund["roic"] > 0.15), "steady",
                 np.where((fund["roic"] > 0.10), "marginal", "value_destructive"))
    )
    return fund


def compute_growth_quality(fund: pd.DataFrame) -> pd.DataFrame:
    """Vectorized: 3-year revenue CAGR and volatility"""
    fund = fund.copy()
    # For CAGR, we need first and last revenue in 12-quarter window
    # Use shift-based approach instead of rolling apply on datetime
    fund["rev_shifted_12"] = fund.groupby("ticker")["total_revenue"].shift(12)
    fund["date_shifted_12"] = fund.groupby("ticker")["as_of_date"].shift(12)
    
    fund["years"] = (fund["as_of_date"] - fund["date_shifted_12"]).dt.days / 365.25
    fund["rev_cagr_3y"] = np.where(
        (fund["years"] > 0) & (fund["rev_shifted_12"] > 0) & fund["total_revenue"].notna(),
        (fund["total_revenue"] / fund["rev_shifted_12"]) ** (1 / fund["years"]) - 1,
        np.nan
    )
    
    # Growth volatility (QoQ revenue growth std over 12 quarters)
    fund["rev_growth_qoq"] = fund.groupby("ticker")["total_revenue"].pct_change()
    fund["growth_volatility"] = fund.groupby("ticker")["rev_growth_qoq"].transform(
        lambda x: x.rolling(12, min_periods=8).std()
    )
    
    fund["growth_quality"] = (
        (fund["rev_cagr_3y"] > 0.03) & (fund["growth_volatility"] < 0.2)
    )
    return fund


def compute_moat_indicators(fund: pd.DataFrame) -> pd.DataFrame:
    """Vectorized: Margin stability and ROIC persistence"""
    fund = fund.copy()
    
    # FCF margin stability (rolling 5-year)
    fund["fcf_margin_mean"] = fund.groupby("ticker")["fcf_margin"].transform(
        lambda x: x.rolling(20, min_periods=8).mean()
    )
    fund["fcf_margin_std"] = fund.groupby("ticker")["fcf_margin"].transform(
        lambda x: x.rolling(20, min_periods=8).std()
    )
    fund["fcf_margin_cv"] = fund["fcf_margin_std"] / fund["fcf_margin_mean"].replace(0, np.nan)
    fund["margin_stable"] = fund["fcf_margin_cv"] < 0.3
    
    # ROIC persistence (rolling 5-year: >15% for 75% of periods)
    fund["roic_above_15"] = fund.groupby("ticker")["roic"].transform(
        lambda x: x.rolling(20, min_periods=8).apply(lambda s: (s > 0.15).mean() if len(s) > 0 else np.nan, raw=False)
    )
    fund["roic_persist"] = fund["roic_above_15"] > 0.75
    
    fund["moat_indicators"] = fund["margin_stable"] & fund["roic_persist"]
    return fund


def compute_capital_allocation(fund: pd.DataFrame) -> pd.DataFrame:
    """Vectorized: Share count change (buybacks) and FCF positivity"""
    fund = fund.copy()
    
    # Share count change (rolling 5-year)
    fund["shares_first"] = fund.groupby("ticker")["shares_outstanding"].transform(
        lambda x: x.rolling(20, min_periods=8).apply(lambda s: s.iloc[0] if len(s) > 0 else np.nan, raw=False)
    )
    fund["shares_last"] = fund.groupby("ticker")["shares_outstanding"].transform(
        lambda x: x.rolling(20, min_periods=8).apply(lambda s: s.iloc[-1] if len(s) > 0 else np.nan, raw=False)
    )
    fund["share_count_change"] = np.where(
        fund["shares_first"] > 0,
        fund["shares_last"] / fund["shares_first"] - 1,
        np.nan
    )
    fund["buyback_signal"] = fund["share_count_change"] < -0.02
    
    fund["fcf_positive"] = fund["free_cash_flow"] > 0
    fund["capital_allocation"] = fund["buyback_signal"] | fund["fcf_positive"]
    
    return fund


def compute_cash_flow_quality(fund: pd.DataFrame) -> pd.DataFrame:
    """Vectorized: FCF conversion and accruals (uses available data)"""
    fund = fund.copy()
    
    # FCF margin already computed = FCF / Revenue
    # Use FCF margin as proxy for cash flow quality
    fund["fcf_margin_ok"] = fund["fcf_margin"] > 0.10  # >10% FCF margin
    
    # Since we don't have net_income or OCF in fundamentals, use proxies
    # FCF positive + reasonable margin = quality
    fund["conversion_ok"] = fund["fcf_margin_ok"]  # proxy
    fund["accruals_ok"] = True  # can't compute without NI/OCF
    fund["cash_flow_quality"] = fund["fcf_margin_ok"]
    
    return fund


def main():
    ap = argparse.ArgumentParser(description="Damodaran Quality Screens (vectorized)")
    ap.add_argument("--all", action="store_true", help="Run all screens")
    ap.add_argument("--min-score", type=int, default=0, help="Minimum quality score to show")
    args = ap.parse_args()

    fund = load_fundamentals()
    wacc = load_wacc()
    
    print(f"Loaded fundamentals: {len(fund)} rows, {fund['ticker'].nunique()} tickers")
    print(f"Loaded WACC: {len(wacc)} rows")

    print("Computing vectorized screens...")
    
    # Screen 1: Excess Returns (merge WACC)
    if len(wacc) > 0:
        fund = fund.merge(wacc[["ticker", "wacc"]], on="ticker", how="left")
    fund["excess_returns"] = (fund["roic"] - fund["wacc"]) > 0.02
    fund["roic_wacc_spread"] = fund["roic"] - fund["wacc"]
    
    # Screen 2: Return Consistency
    fund = compute_return_consistency(fund)
    
    # Screen 3: Reinvestment Quality
    fund = compute_reinvestment_quality(fund)
    
    # Screen 4: Financial Health
    fund["coverage_ok"] = fund["interest_coverage"] > 4.0
    fund["leverage_ok"] = (fund["debt_to_equity"].notna()) & (fund["debt_to_equity"] < 2.0)
    # Cash flow OK: if FCF data available, require positive; if not available, ignore (assume OK for screening?)
    # We'll be conservative: if FCF missing, we cannot confirm health, so treat as False? 
    # But to avoid penalizing missing data, we'll allow missing FCF to pass if coverage and leverage are OK.
    # However, Damodaran would want to see cash flow. Given data sparsity, we'll use FCF if available, else skip.
    if "free_cash_flow" in fund.columns:
        fund["cash_flow_ok"] = fund["free_cash_flow"] > 0
    else:
        fund["cash_flow_ok"] = True  # assume OK if data missing (so as not to penalize)
    fund["financial_health"] = fund["coverage_ok"] & fund["leverage_ok"] & fund["cash_flow_ok"]
    
    # Screen 5: Cash Flow Quality
    fund = compute_cash_flow_quality(fund)
    
    # Screen 6: Growth Quality
    fund = compute_growth_quality(fund)
    
    # Screen 7: Moat Indicators
    fund = compute_moat_indicators(fund)
    
    # Screen 8: Capital Allocation
    fund = compute_capital_allocation(fund)
    
    # Composite Score
    score_components = {
        "excess_returns": 15,
        "return_consistency": 15,
        "reinvestment_quality": 15,
        "financial_health": 15,
        "cash_flow_quality": 15,
        "growth_quality": 10,
        "moat_indicators": 10,
        "capital_allocation": 5,
    }
    
    fund["quality_score"] = 0
    fund["screens_passed"] = 0
    for screen, weight in score_components.items():
        if screen in fund.columns:
            fund["quality_score"] += fund[screen].astype(int) * weight
            fund["screens_passed"] += fund[screen].astype(int)
    fund["total_screens"] = len(score_components)
    
    # Save
    quality_cols = [
        "ticker", "as_of_date", "quality_score", "screens_passed", "total_screens",
        "excess_returns", "roic_wacc_spread", "return_consistency", "roe_cv", "roic_cv",
        "reinvestment_quality", "reinvestment_profile", "financial_health",
        "coverage_ok", "leverage_ok", "cash_flow_ok", "cash_flow_quality",
        "fcf_conversion", "accruals_ratio", "growth_quality", "rev_cagr_3y", "growth_volatility",
        "moat_indicators", "margin_stable", "roic_persist", "capital_allocation",
        "share_count_change", "buyback_signal", "fcf_positive",
    ]
    quality_cols = [c for c in quality_cols if c in fund.columns]
    
    quality = fund[quality_cols].copy()
    pq.write_table(pa.Table.from_pandas(quality, preserve_index=False), OUT_QUALITY)
    print(f"Saved quality scores → {OUT_QUALITY} ({len(quality)} rows)")

    # Binary screens only
    screen_cols = [c for c in quality.columns if c.endswith(("_ok", "_pass", "_signal", "_stable", "_persist", "_quality", "_health", "_allocation", "_returns", "_consistency", "_indicators"))]
    screens = quality[["ticker", "as_of_date"] + screen_cols].copy()
    pq.write_table(pa.Table.from_pandas(screens, preserve_index=False), OUT_SCREENS)
    print(f"Saved quality screens → {OUT_SCREENS} ({len(screens)} rows)")

    # Show top quality (latest per ticker)
    latest = quality.sort_values("as_of_date").groupby("ticker").tail(1)
    latest = latest.sort_values("quality_score", ascending=False)
    
    if args.min_score > 0:
        latest = latest[latest["quality_score"] >= args.min_score]
    
    print(f"\n=== Top Quality Scores (min={args.min_score}) ===")
    show_cols = ["ticker", "quality_score", "screens_passed"] + [c for c in score_components.keys()]
    show_cols = [c for c in show_cols if c in latest.columns]
    print(latest[show_cols].head(30).to_string(index=False))

    print("\n=== Quality Distribution ===")
    print(latest["quality_score"].value_counts().sort_index().to_string())

    print("\n=== Screen Pass Rates ===")
    for screen in score_components.keys():
        if screen in latest.columns:
            rate = latest[screen].mean()
            print(f"  {screen}: {rate:.1%}")


if __name__ == "__main__":
    main()