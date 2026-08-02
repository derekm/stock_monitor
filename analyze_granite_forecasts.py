#!/usr/bin/env python3
"""
analyze_granite_forecasts.py — Summarize Granite forecasts and compare tickers.

Usage:
  python analyze_granite_forecasts.py
  python analyze_granite_forecasts.py --ticker MOS,CF,SHEL
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
FORECAST_CSV = DATA_DIR / "forecasts_granite.csv"
FORECAST_PQ = DATA_DIR / "forecasts_granite.parquet"
BACKTEST_FILE = DATA_DIR / "forecast_backtest_metrics.csv"
PRICES_FILE = DATA_DIR / "daily_prices.parquet"


def load_forecasts() -> pd.DataFrame:
    if FORECAST_PQ.exists():
        try:
            return pd.read_parquet(FORECAST_PQ)
        except Exception:
            pass
    if FORECAST_CSV.exists():
        return pd.read_csv(FORECAST_CSV, parse_dates=["as_of", "forecast_date"])
    raise SystemExit("No forecasts found. Run: python forecast_granite.py forecast --ticker MOS")


def main():
    parser = argparse.ArgumentParser(description="Analyze Granite stock forecasts")
    parser.add_argument("--ticker", help="Filter tickers (comma-separated)")
    parser.add_argument("--index", help="Filter by index_name (comma-separated; substring match per label)")
    args = parser.parse_args()

    fc = load_forecasts()
    if args.ticker:
        tickers = [x.strip().upper() for x in args.ticker.split(",")]
        fc = fc[fc["ticker"].isin(tickers)]
    if args.index and "index_name" in fc.columns:
        wanted = [x.strip().lower() for x in args.index.split(",")]
        def _match(cell):
            labels = [p.strip().lower() for p in str(cell).split(",") if p.strip()]
            return any(w in labels for w in wanted)
        fc = fc[fc["index_name"].map(_match)]

    print("=" * 70)
    print("GRANITE FORECAST SUMMARY")
    print("=" * 70)
    print(f"As-of dates: {fc['as_of'].min()} → {fc['as_of'].max()}")
    print(f"Tickers: {sorted(fc['ticker'].unique())}")
    if "index_name" in fc.columns:
        print(f"Indexes: {sorted({p.strip() for s in fc['index_name'].dropna().astype(str) for p in s.split(',') if p.strip()})}")
    print(f"Horizons: {sorted(fc['horizon'].unique())}")
    print(f"Backend: {fc['backend'].iloc[0] if len(fc) else 'n/a'}")

    # Point forecast table at max horizon
    max_h = fc["horizon"].max()
    tail = fc[fc["horizon"] == max_h].copy()
    tail["signal"] = tail["pct_change"].apply(
        lambda x: "BULL" if x > 3 else ("BEAR" if x < -3 else "NEUTRAL")
    )
    print(f"\n--- Horizon H+{max_h} snapshot ---")
    cols = ["ticker", "last_close", "forecast_close", "pct_change", "signal"]
    print(tail[cols].sort_values("pct_change", ascending=False).to_string(index=False))

    # Path shape: early vs late horizon
    print("\n--- Term structure (mean % change by horizon) ---")
    ts = fc.groupby("horizon")["pct_change"].mean()
    for h, v in ts.items():
        bar = "+" * max(0, int(v)) + "-" * max(0, int(-v))
        print(f"  H+{int(h):02d}  {v:+6.2f}%  {bar}")

    # Per-ticker expected move
    print("\n--- Expected move by ticker (final horizon) ---")
    for _, r in tail.sort_values("pct_change", ascending=False).iterrows():
        print(f"  {r['ticker']:6}  {r['pct_change']:+6.2f}%  ({r['last_close']:.2f} → {r['forecast_close']:.2f})")

    if BACKTEST_FILE.exists():
        bt = pd.read_csv(BACKTEST_FILE)
        if args.ticker:
            bt = bt[bt["ticker"].isin([x.strip().upper() for x in args.ticker.split(",")])]
        if args.index and "index_name" in bt.columns:
            wanted = [x.strip().lower() for x in args.index.split(",")]
            def _match(cell):
                labels = [p.strip().lower() for p in str(cell).split(",") if p.strip()]
                return any(w in labels for w in wanted)
            bt = bt[bt["index_name"].map(_match)]
        print("\n--- Backtest metrics ---")
        print(bt.to_string(index=False))
        if "index_name" in bt.columns and len(bt):
            print("\n--- Backtest by index (mean DirAcc / MAE) ---")
            # explode multi labels for summary
            rows = []
            for _, r in bt.iterrows():
                for lab in str(r.get("index_name", "")).split(","):
                    lab = lab.strip()
                    if lab:
                        rows.append({"index_name": lab, "directional_accuracy": r.get("directional_accuracy"), "mae": r.get("mae")})
            if rows:
                import numpy as np
                sm = pd.DataFrame(rows).groupby("index_name").agg(
                    n=("mae", "count"), mean_mae=("mae", "mean"), mean_diracc=("directional_accuracy", "mean")
                ).reset_index()
                print(sm.to_string(index=False))

    # Optional: vs last realized return for context
    if PRICES_FILE.exists():
        try:
            prices = pd.read_parquet(PRICES_FILE)
            prices["date"] = pd.to_datetime(prices["date"])
            print("\n--- Recent 20d realized vs forecast signal ---")
            for t in sorted(fc["ticker"].unique()):
                s = prices[prices["ticker"] == t].sort_values("date")["close"]
                if len(s) < 21:
                    continue
                realized = (s.iloc[-1] / s.iloc[-21] - 1) * 100
                exp = tail.loc[tail["ticker"] == t, "pct_change"]
                exp = float(exp.iloc[0]) if len(exp) else float("nan")
                print(f"  {t:6}  realized_20d={realized:+6.2f}%  forecast_H{max_h}={exp:+6.2f}%")
        except Exception:
            pass


if __name__ == "__main__":
    main()
