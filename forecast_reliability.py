#!/usr/bin/env python3
"""
forecast_reliability.py — Rank forecast setups on holdings after first trade.

Runs several backtest configurations with --from-first-trade semantics and
writes a comparison table so you can pick more reliable setups.

Usage:
  python forecast_reliability.py --index portfolio --save
  python forecast_reliability.py --ticker MOS,PFE --horizons 5,10,20 --windows 40,60 --save
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
BACKTEST = DATA_DIR / "forecast_backtest_metrics.csv"
OUT = DATA_DIR / "forecast_reliability_rank.csv"


def run_one(tickers: str | None, index: str | None, horizon: int, window: int, context: int) -> pd.DataFrame:
    cmd = [sys.executable, str(DATA_DIR / "forecast_granite.py"), "backtest",
           "--horizon", str(horizon), "--window", str(window), "--context", str(context),
           "--from-first-trade"]
    if tickers:
        cmd += ["--ticker", tickers]
    elif index:
        cmd += ["--index", index]
    else:
        cmd += ["--index", "portfolio"]
    subprocess.run(cmd, cwd=str(DATA_DIR), check=False)
    if not BACKTEST.exists():
        return pd.DataFrame()
    df = pd.read_csv(BACKTEST)
    df["setup_horizon"] = horizon
    df["setup_window"] = window
    df["setup_context"] = context
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--index", default="portfolio")
    ap.add_argument("--horizons", default="5,10,20")
    ap.add_argument("--windows", default="40,60,90")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    frames = []
    for h in horizons:
        for w in windows:
            print(f"\n=== setup H={h} W={w} (from_first_trade) ===")
            df = run_one(args.ticker, args.index, h, w, args.context)
            if len(df):
                frames.append(df)
    if not frames:
        print("No results")
        return
    all_df = pd.concat(frames, ignore_index=True)
    # rank setups by mean directional accuracy then MAE
    summary = (
        all_df.groupby(["setup_horizon", "setup_window"], as_index=False)
        .agg(
            mean_diracc=("directional_accuracy", "mean"),
            mean_mae=("mae", "mean"),
            mean_mape=("mape_pct", "mean"),
            n_tickers=("ticker", "nunique"),
            n_origins=("n_origins", "sum"),
        )
        .sort_values(["mean_diracc", "mean_mae"], ascending=[False, True])
    )
    summary["rank"] = range(1, len(summary) + 1)
    print("\n=== Reliability ranking (higher DirAcc, lower MAE) ===")
    print(summary.to_string(index=False))
    if args.save:
        all_df.to_csv(DATA_DIR / "forecast_reliability_detail.csv", index=False)
        summary.to_csv(OUT, index=False)
        print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
