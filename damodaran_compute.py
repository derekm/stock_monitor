#!/usr/bin/env python3
"""
damodaran_compute.py — Compute Damodaran life cycle, fair multiples, and revenue growth time series.

Inputs:
- fundamentals.parquet (with total_revenue, free_cash_flow, capital_expenditure, fcf_margin, reinvestment_rate)

Outputs:
- revenue_growth.parquet      — ticker × as_of_date × revenue_growth_yoy/qoq
- fcf_margin_history.parquet  — ticker × as_of_date × fcf_margin
- reinvestment_rate_history.parquet — ticker × as_of_date × reinvestment_rate
- life_cycle_stage.parquet    — ticker × as_of_date × life_cycle_stage
- fair_multiples.parquet      — ticker × as_of_date × fair_pe, fair_ev_ebitda, fair_ev_sales, fair_pb
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

OUT_REV_GROWTH = DATA_DIR / "revenue_growth.parquet"
OUT_FCF_MARGIN = DATA_DIR / "fcf_margin_history.parquet"
OUT_REINVEST = DATA_DIR / "reinvestment_rate_history.parquet"
OUT_LIFE_CYCLE = DATA_DIR / "life_cycle_stage.parquet"
OUT_FAIR_MULT = DATA_DIR / "fair_multiples.parquet"


def load_fundamentals() -> pd.DataFrame:
    if not FUND.exists():
        raise FileNotFoundError(f"{FUND} not found")
    df = pd.read_parquet(FUND)
    if "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df = df.sort_values(["ticker", "as_of_date"])
    return df


def compute_revenue_growth(fund: pd.DataFrame) -> pd.DataFrame:
    """Compute YoY and QoQ revenue growth per ticker (only on real quarterly data)."""
    results = []
    for ticker, g in fund.groupby("ticker"):
        g = g.sort_values("as_of_date").copy()
        
        # Filter to rows with actual revenue data
        real = g[g["revenue_quarterly"].notna()].copy()
        if len(real) < 4:
            # Not enough data for TTM
            out = g[["ticker", "as_of_date", "revenue_quarterly"]].copy()
            out["rev_ttm"] = np.nan
            out["revenue_growth_yoy"] = np.nan
            out["revenue_growth_qoq"] = np.nan
            out["revenue_growth"] = np.nan
            results.append(out)
            continue
        
        # Compute TTM and growth on real quarterly data
        real = real.sort_values("as_of_date")
        real["rev_ttm"] = real["revenue_quarterly"].rolling(4, min_periods=4).sum()
        real["revenue_growth_yoy"] = real["rev_ttm"].pct_change(4)
        real["revenue_growth_qoq"] = real["revenue_quarterly"].pct_change(1)
        real["revenue_growth"] = real["revenue_growth_yoy"]
        
        # Merge back to full frame
        out = g[["ticker", "as_of_date", "revenue_quarterly"]].copy()
        out = out.merge(
            real[["ticker", "as_of_date", "rev_ttm", "revenue_growth_yoy", "revenue_growth_qoq", "revenue_growth"]],
            on=["ticker", "as_of_date"], how="left"
        )
        results.append(out)
    
    return pd.concat(results, ignore_index=True)


def compute_life_cycle(fund: pd.DataFrame) -> pd.DataFrame:
    """Classify corporate life cycle per Damodaran 6-stage framework.
    
    Vectorized — operates on entire DataFrame at once.
    """
    # Check if revenue_growth already exists in fund
    if "revenue_growth" not in fund.columns:
        # First compute revenue growth
        rev_growth = compute_revenue_growth(fund)
        # Merge with original fund data
        fund = fund.merge(
            rev_growth[["ticker", "as_of_date", "revenue_growth"]],
            on=["ticker", "as_of_date"], how="left"
        )
    
    # Vectorized classification
    rev_g = fund["revenue_growth"]
    fcf_margin = fund["fcf_margin"]
    reinvest = fund["reinvestment_rate"]
    roic = fund["roic"]
    
    # Default: Unclassified
    stage = pd.Series("Unclassified", index=fund.index)
    
    # Decline
    stage = np.where(rev_g.notna() & (rev_g < -0.05), "Decline", stage)
    
    # Young Growth: high growth, negative FCF margin
    stage = np.where(
        rev_g.notna() & (rev_g > 0.25) & fcf_margin.notna() & (fcf_margin < 0),
        "Young Growth", stage
    )
    
    # High Growth: high growth, positive FCF margin
    stage = np.where(
        rev_g.notna() & (rev_g > 0.25) & fcf_margin.notna() & (fcf_margin >= 0),
        "High Growth", stage
    )
    
    # High Growth: moderate-high growth, low FCF margin
    stage = np.where(
        rev_g.notna() & (rev_g > 0.15) & (rev_g <= 0.25) & fcf_margin.notna() & (fcf_margin < 0.10),
        "High Growth", stage
    )
    
    # Mature Growth: moderate growth, higher FCF margin
    stage = np.where(
        rev_g.notna() & (rev_g > 0.15) & (rev_g <= 0.25) & fcf_margin.notna() & (fcf_margin >= 0.10),
        "Mature Growth", stage
    )
    
    # Mature Growth: 5-15% growth
    stage = np.where(
        rev_g.notna() & (rev_g > 0.05) & (rev_g <= 0.15), "Mature Growth", stage
    )
    
    # Mature Stable: 0-5% growth, high FCF margin
    stage = np.where(
        rev_g.notna() & (rev_g > 0) & (rev_g <= 0.05) & fcf_margin.notna() & (fcf_margin > 0.10),
        "Mature Stable", stage
    )
    
    # Mature Stable: 0-5% growth, low FCF margin
    stage = np.where(
        rev_g.notna() & (rev_g > 0) & (rev_g <= 0.05) & fcf_margin.notna() & (fcf_margin <= 0.10),
        "Mature Growth", stage
    )
    
    # Mature Stable: negative growth but positive FCF margin
    stage = np.where(
        rev_g.notna() & (rev_g <= 0) & (rev_g >= -0.05) & fcf_margin.notna() & (fcf_margin > 0.10),
        "Mature Stable", stage
    )
    
    results = fund[["ticker", "as_of_date"]].copy()
    results["life_cycle_stage"] = stage
    results["revenue_growth"] = rev_g.values
    results["fcf_margin"] = fcf_margin.values
    results["reinvestment_rate"] = reinvest.values
    results["roic"] = roic.values
    
    return results


def compute_fair_multiples(fund: pd.DataFrame, erp: float = 0.0423, rf: float = 0.04) -> pd.DataFrame:
    """Compute Damodaran fundamental-implied fair multiples per ticker/date.
    
    Vectorized implementation using numpy broadcasting.
    
    Fair P/E = (1 - reinvestment_rate) * (1 + g) / (cost_of_equity - g)
    Fair EV/EBITDA = (1 - tax_rate) * (1 - reinvestment_rate) * (1 + g) / (wacc - g)
    Fair EV/Sales = Fair EV/EBITDA * (EBITDA/Sales)
    Fair P/B = ROE * (1 - reinvestment_rate) * (1 + g) / (cost_of_equity - g)
    
    Where g = revenue_growth (capped at long-term GDP growth ~2-3%)
    """
    # Check if revenue_growth already exists in fund
    if "revenue_growth" not in fund.columns:
        # First compute revenue growth
        rev_growth = compute_revenue_growth(fund)
        # Merge with original fund data
        fund = fund.merge(
            rev_growth[["ticker", "as_of_date", "revenue_growth"]],
            on=["ticker", "as_of_date"], how="left"
        )
    
    # Load WACC data if available
    from damodaran_data import WACC_PER_TICKER
    wacc_df = pd.read_parquet(WACC_PER_TICKER) if WACC_PER_TICKER.exists() else pd.DataFrame()
    if len(wacc_df):
        wacc_df = wacc_df.set_index("ticker")
        fund = fund.merge(
            wacc_df[["wacc", "cost_of_equity"]].rename(columns={"wacc": "_wacc", "cost_of_equity": "_coe"}),
            on="ticker", how="left"
        )
    else:
        fund["_wacc"] = 0.09
        fund["_coe"] = 0.09
    
    # Vectorized fair multiple calculation
    reinvest = fund["reinvestment_rate"]
    cost_of_equity = fund["_coe"]
    wacc = fund["_wacc"]
    roe = fund["roe"]
    ev_ebitda = fund["ev_ebitda"]
    
    # Long-term growth cap
    g_long = fund["revenue_growth"].clip(lower=0, upper=0.03)
    g_long = g_long.fillna(0.02)
    
    tax_rate = 0.25
    
    # Extract columns as Series
    reinvest = fund["reinvestment_rate"]
    roic = fund["roic"]
    roe = fund["roe"]
    wacc_val = fund["_wacc"]
    cost_of_equity = fund["_coe"]
    
    # Use reinvestment_rate if available, otherwise compute from g/ROIC or g/ROE
    reinvest_calc = reinvest.copy()
    mask_no_reinvest = reinvest_calc.isna() & roic.notna() & (roic > 0)
    reinvest_calc[mask_no_reinvest] = (g_long / roic)[mask_no_reinvest]
    mask_still_missing = reinvest_calc.isna() & roe.notna() & (roe > 0)
    reinvest_calc[mask_still_missing] = (g_long / roe)[mask_still_missing]
    reinvest_calc = reinvest_calc.fillna(0.5)
    
    # Fair P/E: payout / (cost_of_equity - g)
    fair_pe = pd.Series(np.nan, index=fund.index)
    pe_mask = cost_of_equity.notna() & (cost_of_equity > g_long) & roe.notna() & (roe > 0)
    payout = np.clip(1 - g_long / roe, 0, 1)
    fair_pe = np.where(pe_mask, payout / (cost_of_equity - g_long), fair_pe)
    
    # Fair P/B: (ROE - g) / (cost_of_equity - g)
    fair_pb = pd.Series(np.nan, index=fund.index)
    pb_mask = pe_mask
    fair_pb = np.where(pb_mask, (roe - g_long) / (cost_of_equity - g_long), fair_pb)
    
    # Fair EV/EBITDA: (1 - reinvest) * (1 - t) / (wacc - g)
    fair_ev_ebitda = pd.Series(np.nan, index=fund.index)
    ev_mask = wacc_val.notna() & (wacc_val > g_long)
    fcf_conversion = (1 - reinvest_calc) * (1 - tax_rate)
    fair_ev_ebitda = np.where(ev_mask, fcf_conversion / (wacc_val - g_long), fair_ev_ebitda)
    
    # Fair EV/Sales: Fair EV/EBITDA * EBITDA margin
    fair_ev_sales = pd.Series(np.nan, index=fund.index)
    sales_mask = ev_mask & ev_ebitda.notna() & (ev_ebitda > 0)
    ebitda_margin = pd.Series(np.nan, index=fund.index)
    ebitda_margin = np.where(sales_mask, 1 / ev_ebitda, ebitda_margin)
    fair_ev_sales = np.where(sales_mask, fair_ev_ebitda * ebitda_margin, fair_ev_sales)
    
    results = fund[["ticker", "as_of_date"]].copy()
    results["fair_pe"] = np.where(np.isfinite(fair_pe), fair_pe, np.nan)
    results["fair_ev_ebitda"] = np.where(np.isfinite(fair_ev_ebitda), fair_ev_ebitda, np.nan)
    results["fair_ev_sales"] = np.where(np.isfinite(fair_ev_sales), fair_ev_sales, np.nan)
    results["fair_pb"] = np.where(np.isfinite(fair_pb), fair_pb, np.nan)
    results["implied_growth"] = g_long
    results["wacc_used"] = wacc
    results["cost_of_equity_used"] = cost_of_equity
    
    return results


def main():
    ap = argparse.ArgumentParser(description="Compute Damodaran time-series metrics")
    ap.add_argument("--all", action="store_true", help="Run all computations")
    ap.add_argument("--revenue-growth", action="store_true")
    ap.add_argument("--fcf-margin", action="store_true")
    ap.add_argument("--reinvestment", action="store_true")
    ap.add_argument("--life-cycle", action="store_true")
    ap.add_argument("--fair-multiples", action="store_true")
    args = ap.parse_args()

    run_all = args.all or not any([
        args.revenue_growth, args.fcf_margin, args.reinvestment,
        args.life_cycle, args.fair_multiples
    ])
    
    fund = load_fundamentals()
    print(f"Loaded fundamentals: {len(fund)} rows, {fund['ticker'].nunique()} tickers")

    if run_all or args.revenue_growth:
        print("Computing revenue growth...")
        rev_growth = compute_revenue_growth(fund)
        # Merge back into fundamentals for downstream use
        fund = fund.merge(
            rev_growth[["ticker", "as_of_date", "revenue_growth", "revenue_growth_yoy", "revenue_growth_qoq"]],
            on=["ticker", "as_of_date"], how="left"
        )
        pq.write_table(pa.Table.from_pandas(rev_growth, preserve_index=False), OUT_REV_GROWTH)
        print(f"  Wrote {OUT_REV_GROWTH} ({len(rev_growth)} rows)")

    if run_all or args.fcf_margin:
        print("Extracting FCF margin history...")
        fcf_hist = fund[["ticker", "as_of_date", "fcf_margin"]].dropna(subset=["fcf_margin"]).copy()
        pq.write_table(pa.Table.from_pandas(fcf_hist, preserve_index=False), OUT_FCF_MARGIN)
        print(f"  Wrote {OUT_FCF_MARGIN} ({len(fcf_hist)} rows)")

    if run_all or args.reinvestment:
        print("Extracting reinvestment rate history...")
        reinvest_hist = fund[["ticker", "as_of_date", "reinvestment_rate"]].dropna(subset=["reinvestment_rate"]).copy()
        pq.write_table(pa.Table.from_pandas(reinvest_hist, preserve_index=False), OUT_REINVEST)
        print(f"  Wrote {OUT_REINVEST} ({len(reinvest_hist)} rows)")

    if run_all or args.life_cycle:
        print("Computing life cycle stage...")
        life_cycle = compute_life_cycle(fund)
        pq.write_table(pa.Table.from_pandas(life_cycle, preserve_index=False), OUT_LIFE_CYCLE)
        print(f"  Wrote {OUT_LIFE_CYCLE} ({len(life_cycle)} rows)")

    if run_all or args.fair_multiples:
        print("Computing fair multiples...")
        fair_mult = compute_fair_multiples(fund)
        pq.write_table(pa.Table.from_pandas(fair_mult, preserve_index=False), OUT_FAIR_MULT)
        print(f"  Wrote {OUT_FAIR_MULT} ({len(fair_mult)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()