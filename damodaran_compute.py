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
        real = g[g["total_revenue"].notna()].copy()
        if len(real) < 4:
            # Not enough data for TTM
            out = g[["ticker", "as_of_date", "total_revenue"]].copy()
            out["rev_ttm"] = np.nan
            out["revenue_growth_yoy"] = np.nan
            out["revenue_growth_qoq"] = np.nan
            out["revenue_growth"] = np.nan
            results.append(out)
            continue
        
        # Compute TTM and growth on real quarterly data
        real = real.sort_values("as_of_date")
        real["rev_ttm"] = real["total_revenue"].rolling(4, min_periods=4).sum()
        real["revenue_growth_yoy"] = real["rev_ttm"].pct_change(4)
        real["revenue_growth_qoq"] = real["total_revenue"].pct_change(1)
        real["revenue_growth"] = real["revenue_growth_yoy"]
        
        # Merge back to full frame
        out = g[["ticker", "as_of_date", "total_revenue"]].copy()
        out = out.merge(
            real[["ticker", "as_of_date", "rev_ttm", "revenue_growth_yoy", "revenue_growth_qoq", "revenue_growth"]],
            on=["ticker", "as_of_date"], how="left"
        )
        results.append(out)
    
    return pd.concat(results, ignore_index=True)


def compute_life_cycle(fund: pd.DataFrame) -> pd.DataFrame:
    """Classify corporate life cycle per Damodaran 6-stage framework.
    
    Stages (Damodaran):
    1. Start-up: negative/low revenue, negative earnings, high reinvestment
    2. Young Growth: high revenue growth (>25%), low/negative margins, high reinvestment
    3. High Growth: high revenue growth (15-25%), improving margins, high reinvestment
    4. Mature Growth: moderate revenue growth (5-15%), positive margins, moderate reinvestment
    5. Mature Stable: low revenue growth (0-5%), stable margins, low reinvestment
    6. Decline: negative revenue growth, declining margins, negative/low reinvestment
    """
    # First compute revenue growth
    rev_growth = compute_revenue_growth(fund)
    # Merge with original fund data to get fcf_margin, reinvestment_rate, roic
    fund = fund.merge(
        rev_growth[["ticker", "as_of_date", "revenue_growth"]],
        on=["ticker", "as_of_date"], how="left"
    )
    
    results = []
    for ticker, g in fund.groupby("ticker"):
        g = g.sort_values("as_of_date").copy()
        
        for _, row in g.iterrows():
            rev_g = row.get("revenue_growth")
            fcf_margin = row.get("fcf_margin")
            reinvest = row.get("reinvestment_rate")
            roic = row.get("roic")
            
            stage = "Unclassified"
            
            if pd.notna(rev_g) and pd.notna(fcf_margin) and pd.notna(reinvest):
                if rev_g < -0.05:
                    stage = "Decline"
                elif rev_g > 0.25:
                    if fcf_margin < 0:
                        stage = "Young Growth"
                    else:
                        stage = "High Growth"
                elif rev_g > 0.15:
                    if fcf_margin < 0.10:
                        stage = "High Growth"
                    else:
                        stage = "Mature Growth"
                elif rev_g > 0.05:
                    stage = "Mature Growth"
                elif rev_g > 0:
                    if fcf_margin > 0.15 and reinvest < 0.3:
                        stage = "Mature Stable"
                    else:
                        stage = "Mature Growth"
                else:
                    stage = "Mature Stable"
            
            results.append({
                "ticker": ticker,
                "as_of_date": row["as_of_date"],
                "life_cycle_stage": stage,
                "revenue_growth": rev_g,
                "fcf_margin": fcf_margin,
                "reinvestment_rate": reinvest,
                "roic": roic,
            })
    
    return pd.DataFrame(results)


def compute_fair_multiples(fund: pd.DataFrame, erp: float = 0.0423, rf: float = 0.04) -> pd.DataFrame:
    """Compute Damodaran fundamental-implied fair multiples per ticker/date.
    
    Fair P/E = (1 - reinvestment_rate) * (1 + g) / (cost_of_equity - g)
    Fair EV/EBITDA = (1 - tax_rate) * (1 - reinvestment_rate) * (1 + g) / (wacc - g)
    Fair EV/Sales = Fair EV/EBITDA * (EBITDA/Sales)
    Fair P/B = ROE * (1 - reinvestment_rate) * (1 + g) / (cost_of_equity - g)
    
    Where g = revenue_growth (capped at long-term GDP growth ~2-3%)
    """
    # First compute revenue growth
    rev_growth = compute_revenue_growth(fund)
    # Merge with original fund data
    fund = fund.merge(
        rev_growth[["ticker", "as_of_date", "revenue_growth"]],
        on=["ticker", "as_of_date"], how="left"
    )
    
    results = []
    for ticker, g in fund.groupby("ticker"):
        g = g.sort_values("as_of_date").copy()
        
        for _, row in g.iterrows():
            rev_g = row.get("revenue_growth")
            fcf_margin = row.get("fcf_margin")
            reinvest = row.get("reinvestment_rate")
            roic = row.get("roic")
            roe = row.get("roe")
            ev_ebitda = row.get("ev_ebitda")
            
            # Need WACC - load from wacc_per_ticker if available
            wacc = row.get("wacc", 0.09)  # default
            cost_of_equity = row.get("cost_of_equity", 0.09)
            tax_rate = 0.25
            
            # Long-term growth cap
            g_long = min(max(rev_g, 0) if pd.notna(rev_g) else 0.02, 0.03)
            
            fair_pe = np.nan
            fair_ev_ebitda = np.nan
            fair_ev_sales = np.nan
            fair_pb = np.nan
            
            if pd.notna(reinvest) and pd.notna(cost_of_equity) and cost_of_equity > g_long:
                # Fair P/E
                fair_pe = (1 - reinvest) * (1 + g_long) / (cost_of_equity - g_long)
                
                # Fair P/B
                if pd.notna(roe) and roe > 0:
                    fair_pb = roe * (1 - reinvest) * (1 + g_long) / (cost_of_equity - g_long)
            
            if pd.notna(reinvest) and pd.notna(wacc) and wacc > g_long:
                # Fair EV/EBITDA
                fair_ev_ebitda = (1 - tax_rate) * (1 - reinvest) * (1 + g_long) / (wacc - g_long)
                
                # Fair EV/Sales ≈ Fair EV/EBITDA * EBITDA margin
                # EBITDA margin ≈ 1/ev_ebitda if we have it
                if pd.notna(ev_ebitda) and ev_ebitda > 0:
                    ebitda_margin = 1 / ev_ebitda
                    fair_ev_sales = fair_ev_ebitda * ebitda_margin
            
            results.append({
                "ticker": ticker,
                "as_of_date": row["as_of_date"],
                "fair_pe": fair_pe if not np.isnan(fair_pe) else None,
                "fair_ev_ebitda": fair_ev_ebitda if not np.isnan(fair_ev_ebitda) else None,
                "fair_ev_sales": fair_ev_sales if not np.isnan(fair_ev_sales) else None,
                "fair_pb": fair_pb if not np.isnan(fair_pb) else None,
                "implied_growth": g_long,
                "wacc_used": wacc,
                "cost_of_equity_used": cost_of_equity,
            })
    
    return pd.DataFrame(results)


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