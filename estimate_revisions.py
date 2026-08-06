#!/usr/bin/env python3
"""
estimate_revisions.py — Consensus EPS-estimate and price-target snapshots with
revision tracking.

Why it exists: the architecture TODO "estimate revisions". Analyst consensus
changes are a leading fundamental signal. yfinance exposes current consensus
(earnings_estimate, analyst_price_targets) but no history — so this script
SNAPSHOTS the current consensus into estimate_revisions.parquet on each run
(append), and the revision columns compare the latest snapshot against the
previous one (pct change in mean EPS estimate / mean price target).

First run just seeds the baseline; subsequent runs (daily automation) produce
revisions. Run at least twice with a gap to see meaningful revision data.

Output:
  estimate_revisions.parquet — long table: (snapshot_date, ticker, period,
    mean_eps, mean_eps_rev_pct, mean_pt, mean_pt_rev_pct)

Usage:
    python estimate_revisions.py [--save] [--tickers AAPL,MSFT]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from analytics_common import DATA_DIR

OUT = DATA_DIR / "estimate_revisions.parquet"


def snapshot_ticker(t: str) -> list[dict]:
    tk = yf.Ticker(t)
    rows = []
    try:
        ee = tk.earnings_estimate
        if ee is not None and not ee.empty and "avg" in ee.columns:
            for period in ee.index:
                try:
                    rows.append({
                        "ticker": t, "period": str(period),
                        "mean_eps": float(ee.loc[period, "avg"]),
                    })
                except Exception:
                    continue
    except Exception:
        pass
    try:
        pt = tk.analyst_price_targets
        if pt is not None and isinstance(pt, dict):
            mean_pt = pt.get("mean")
            if mean_pt is not None and np.isfinite(float(mean_pt)):
                rows.append({"ticker": t, "period": "price_target",
                             "mean_pt": float(mean_pt)})
    except Exception:
        pass
    return rows


def build(tickers: list[str]) -> pd.DataFrame:
    snap = pd.Timestamp.now().normalize()
    rows = []
    for t in tickers:
        rows += snapshot_ticker(t)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    df["snapshot_date"] = snap.date()
    return df


def merge_and_revise(new_df: pd.DataFrame) -> pd.DataFrame:
    if OUT.exists():
        prev = pd.read_parquet(OUT)
    else:
        prev = pd.DataFrame()
    combined = pd.concat([prev, new_df], ignore_index=True)
    if not combined.empty:
        combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"]).dt.date
        # revisions vs previous snapshot for the same (ticker, period)
        combined = combined.sort_values(["ticker", "period", "snapshot_date"])
        g = combined.groupby(["ticker", "period"])["mean_eps"]
        combined["prev_eps"] = g.shift(1)
        combined["mean_eps_rev_pct"] = np.where(
            combined["prev_eps"].notna() & (combined["prev_eps"] != 0),
            (combined["mean_eps"] - combined["prev_eps"]) / combined["prev_eps"].abs() * 100,
            np.nan,
        )
        g2 = combined.groupby(["ticker", "period"])["mean_pt"]
        combined["prev_pt"] = g2.shift(1)
        combined["mean_pt_rev_pct"] = np.where(
            combined["prev_pt"].notna() & (combined["prev_pt"] != 0),
            (combined["mean_pt"] - combined["prev_pt"]) / combined["prev_pt"].abs() * 100,
            np.nan,
        )
    return combined


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=None, help="comma list; default = monitored universe")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        stocks = DATA_DIR / "monitored_stocks.parquet"
        tickers = sorted(pd.read_parquet(stocks)["ticker"].astype(str).str.upper().unique()) if stocks.exists() else []
    new_df = build(tickers)
    if new_df.empty:
        print("No estimate data fetched.")
        return
    out = merge_and_revise(new_df)
    latest = out[out["snapshot_date"] == pd.Timestamp(new_df["snapshot_date"].iloc[0]).date()]
    print(f"=== Estimate revisions snapshot ({len(latest)} rows, {len(out)} total) ===")
    cols = [c for c in ["ticker", "period", "mean_eps", "mean_eps_rev_pct", "mean_pt", "mean_pt_rev_pct"] if c in latest]
    rev = latest[latest.get("mean_eps_rev_pct", pd.Series(dtype=float)).notna()]
    print(latest[cols].head(15).to_string(index=False))
    if args.save:
        out.to_parquet(OUT, index=False)
        print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
