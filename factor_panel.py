#!/usr/bin/env python3
"""factor_panel.py — multi-factor panel: value, quality, momentum, low-vol, leverage flag.

Combines preferred_metrics + momentum_metrics into a single scoring panel and
simple equal-risk-contribution style rank composite.

Usage:
  python factor_panel.py --save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
PREF = DATA_DIR / "preferred_metrics.parquet"
MOM = DATA_DIR / "momentum_metrics.parquet"
OUT = DATA_DIR / "factor_panel.parquet"
OUT_TOP = DATA_DIR / "factor_panel_top.parquet"


def z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    if not sd or not np.isfinite(sd):
        return s * 0.0
    return (s - s.mean()) / sd


def build() -> pd.DataFrame:
    pref = pd.read_parquet(PREF) if PREF.exists() else pd.DataFrame()
    mom = pd.read_parquet(MOM) if MOM.exists() else pd.DataFrame()
    if pref.empty:
        return pd.DataFrame()
    df = pref.copy()
    if len(mom):
        df = df.merge(mom[["ticker", "momentum_score", "mom_12_1", "ret_63d", "resid_mom_63", "momentum_quintile"]],
                      on="ticker", how="left")
    # factor z-scores
    df["f_value"] = z(1.0 / df["ev_ebitda"].replace(0, np.nan)) if "ev_ebitda" in df.columns else np.nan
    if "pb_ratio" in df.columns:
        df["f_value"] = 0.5 * df["f_value"].fillna(0) + 0.5 * z(1.0 / df["pb_ratio"].replace(0, np.nan)).fillna(0)
    df["f_quality"] = 0.5 * z(df.get("roe")).fillna(0) + 0.5 * z(df.get("roic")).fillna(0)
    df["f_momentum"] = z(df["momentum_score"]) if "momentum_score" in df.columns else 0.0
    df["f_lowvol"] = z(1.0 / df["name_vol"].replace(0, np.nan)) if "name_vol" in df.columns else 0.0
    # leverage penalty
    df["f_leverage_pen"] = 0.0
    if "leverage_flag" in df.columns:
        df.loc[df["leverage_flag"] == "levered-assets", "f_leverage_pen"] = -0.5
        df.loc[df["leverage_flag"] == "cheap-assets", "f_leverage_pen"] = 0.15
    # composite factor score
    df["factor_composite"] = (
        0.25 * df["f_value"].fillna(0)
        + 0.25 * df["f_quality"].fillna(0)
        + 0.25 * df["f_momentum"].fillna(0)
        + 0.15 * df["f_lowvol"].fillna(0)
        + 0.10 * df["f_leverage_pen"].fillna(0)
    )
    df = df.sort_values("factor_composite", ascending=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    df = build()
    cols = [c for c in (
        "ticker", "decision", "factor_composite", "f_value", "f_quality", "f_momentum", "f_lowvol",
        "momentum_score", "leverage_flag", "composite_score", "roe", "ev_ebitda"
    ) if c in df.columns]
    print(df[cols].head(15).to_string(index=False))
    if args.save and len(df):
        df.to_parquet(OUT)
        df.head(25)[cols].to_parquet(OUT_TOP)
        print(f"Wrote {OUT.name}, {OUT_TOP.name}")


if __name__ == "__main__":
    main()
