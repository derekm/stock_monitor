#!/usr/bin/env python3
"""rebalance_calendar.py — Regime- and dual-pass-aware rebalance schedule.

Rules (defaults):
  - Monthly: last trading day of month (from daily_prices calendar)
  - Skip / reduce if current regime is high_vol_stress (optional half-band)
  - Accelerate (mid-month) if dual-pass set churns heavily

Usage:
  python rebalance_calendar.py --months 12 --save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "rebalance_calendar.csv"
PRICES = DATA_DIR / "daily_prices.parquet"
HMM = DATA_DIR / "hmm_regimes.csv"
PREF = DATA_DIR / "preferred_metrics.csv"


def trading_calendar() -> pl.DataFrame:
    d = (
        pl.scan_parquet(str(PRICES))
        .select(pl.col("date").cast(pl.Date, strict=False))
        .unique()
        .sort("date")
        .collect()
    )
    return d


def month_end_dates(cal: pl.DataFrame, months: int) -> list:
    df = cal.with_columns([
        pl.col("date").dt.year().alias("y"),
        pl.col("date").dt.month().alias("m"),
    ])
    ends = df.group_by(["y", "m"]).agg(pl.col("date").max().alias("date")).sort("date")
    return ends.tail(months)["date"].to_list()


def latest_regime_on(date) -> str:
    if not HMM.exists():
        return "unknown"
    try:
        h = pd.read_csv(HMM)
        if "date" not in h.columns:
            return "unknown"
        h["date"] = pd.to_datetime(h["date"])
        h = h[h["date"] <= pd.Timestamp(date)]
        if h.empty:
            return "unknown"
        row = h.iloc[-1]
        for c in ("regime", "state", "label"):
            if c in h.columns:
                return str(row[c])
        return "unknown"
    except Exception:
        return "unknown"


def dual_core_tickers() -> set[str]:
    if not PREF.exists():
        return set()
    df = pd.read_csv(PREF)
    if "decision" in df.columns:
        return set(df.loc[df["decision"] == "INCLUDE_CORE", "ticker"].astype(str))
    return set()


def build(months: int = 12) -> pd.DataFrame:
    cal = trading_calendar()
    ends = month_end_dates(cal, months)
    core = dual_core_tickers()
    rows = []
    for d in ends:
        regime = latest_regime_on(d)
        action = "full_rebalance"
        band = 1.0
        note = "month-end"
        if "stress" in regime.lower() or regime == "high_vol_stress":
            action = "reduced_rebalance"
            band = 0.5
            note = "high_vol_stress → half turnover band"
        rows.append({
            "rebalance_date": pd.Timestamp(d).date().isoformat(),
            "regime": regime,
            "action": action,
            "turnover_band": band,
            "n_dual_core": len(core),
            "notes": note,
        })
    # next scheduled mid-month review if stress
    out = pd.DataFrame(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    df = build(args.months)
    print(df.tail(12).to_string(index=False))
    if args.save:
        df.to_csv(OUT, index=False)
        print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
