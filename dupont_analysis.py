#!/usr/bin/env python3
"""
dupont_analysis.py — DuPont decomposition of ROE.

Classic 3-step DuPont:
  ROE = Profit Margin (NI/Sales) × Asset Turnover (Sales/Assets) × Equity Multiplier (Assets/Equity)

If NI/Sales and Sales not available, estimate from ROE + D/E:
  Equity Multiplier ≈ 1 + D/E
  Then PM × AT = ROE / EM  (profitability × efficiency residual)

Buffett lens: prefers high ROE driven by **margin and turnover**, not leverage (high EM).

Usage:
  python dupont_analysis.py
  python dupont_analysis.py --min-roe 0.15 --save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
OUT = DATA_DIR / "dupont_analysis.parquet"


def dupont_row(r: pd.Series) -> dict:
    roe = r.get("roe")
    de = r.get("debt_to_equity")
    roic = r.get("roic")
    # Equity multiplier from D/E
    if pd.notna(de) and de >= 0:
        em = 1.0 + float(de)
    else:
        em = np.nan
    # Residual profitability×efficiency
    if pd.notna(roe) and pd.notna(em) and em > 0:
        pm_at = float(roe) / em
    else:
        pm_at = np.nan
    # Leverage contribution share
    if pd.notna(roe) and roe != 0 and pd.notna(em):
        # unlevered-ish proxy
        leverage_boost = float(roe) - pm_at if pd.notna(pm_at) else np.nan
        lev_share = leverage_boost / float(roe) if pd.notna(leverage_boost) else np.nan
    else:
        leverage_boost = np.nan
        lev_share = np.nan

    # Quality of ROE: high ROE with low leverage is best
    if pd.notna(roe) and pd.notna(em):
        if roe >= 0.15 and em <= 1.5:
            roe_quality = "high_ops"
        elif roe >= 0.15 and em > 2.0:
            roe_quality = "leverage_driven"
        elif roe >= 0.15:
            roe_quality = "mixed"
        elif roe < 0:
            roe_quality = "negative"
        else:
            roe_quality = "modest"
    else:
        roe_quality = "unknown"

    # Align flag language with preferred_metrics leverage_flag
    if pd.notna(em) and em > 2.0 and pd.notna(roe) and roe >= 0.15:
        lev_flag = "levered-assets"
    elif pd.notna(em) and em <= 1.5 and pd.notna(roe) and roe >= 0.15:
        lev_flag = "cheap-assets"  # ops-driven ROE, light leverage
    elif pd.notna(em) and em > 1.5:
        lev_flag = "mixed-assets"
    else:
        lev_flag = ""

    return {
        "roe": roe,
        "roic": roic,
        "debt_to_equity": de,
        "debt_to_assets": round(float(de) / (1.0 + float(de)), 3) if pd.notna(de) and float(de) >= 0 else np.nan,
        "equity_multiplier": round(em, 3) if pd.notna(em) else np.nan,
        "pm_x_at": round(pm_at, 4) if pd.notna(pm_at) else np.nan,
        "leverage_boost": round(leverage_boost, 4) if pd.notna(leverage_boost) else np.nan,
        "leverage_share_of_roe": round(lev_share, 3) if pd.notna(lev_share) else np.nan,
        "roe_quality": roe_quality,
        "leverage_flag": lev_flag,
        "dupont_identity": "ROE = PM × AT × EM",
    }


def run(min_roe: float = 0.0, save: bool = True) -> pd.DataFrame:
    fund = pd.read_parquet(FUND)
    if "as_of_date" in fund.columns:
        fund = fund.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)

    rows = []
    for _, r in fund.iterrows():
        d = dupont_row(r)
        d["ticker"] = r["ticker"]
        d["ev_ebitda"] = r.get("ev_ebitda")
        d["pb_ratio"] = r.get("pb_ratio")
        d["mktcap_to_assets"] = r.get("mktcap_to_assets")
        rows.append(d)
    df = pd.DataFrame(rows)
    if min_roe:
        df = df[df["roe"].fillna(0) >= min_roe]
    df = df.sort_values("roe", ascending=False)

    print("=== DuPont-style ROE drivers ===")
    print("ROE ≈ (PM × Asset Turnover) × Equity Multiplier;  EM ≈ 1 + D/E\n")
    show = ["ticker", "roe", "roic", "equity_multiplier", "pm_x_at", "leverage_share_of_roe", "roe_quality"]
    print(df[show].head(25).to_string(index=False))

    print("\n=== ROE quality mix ===")
    print(df["roe_quality"].value_counts().to_string())

    print("\n=== High ROE but leverage-driven (Buffett wary) ===")
    lev = df[df["roe_quality"] == "leverage_driven"]
    print(lev[show].head(15).to_string(index=False) if len(lev) else "  (none)")

    print("\n=== High ROE from operations (EM≤1.5) — preferred ===")
    ops = df[df["roe_quality"] == "high_ops"]
    print(ops[show].head(15).to_string(index=False) if len(ops) else "  (none)")

    if save:
        df.to_parquet(OUT)
        print(f"\nWrote {OUT}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-roe", type=float, default=0.0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(min_roe=args.min_roe, save=True)


if __name__ == "__main__":
    main()
