#!/usr/bin/env python3
"""
update_polygon.py — Daily OHLCV ingest from Polygon.io (production-grade
price feed), key-gated.

Why it exists: the architecture TODO "integrate data sources: Polygon
(production)". The repo ingests via yfinance (prototyping); Polygon is the
production alternative. This script pulls daily bars for the monitored
universe and appends them into daily_prices.parquet (source='polygon').

Key-gated by design: requires POLYGON_API_KEY env var. Without it, the
script explains how to get a key and exits 0 (no crash in the automation).
Fills the polygon free-tier bars endpoint (5 requests/sec limit respected).

Usage:
    export POLYGON_API_KEY=...
    python update_polygon.py [--days 5] [--save]
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from analytics_common import DATA_DIR

PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
BASE = "https://api.polygon.io/v2/aggs/ticker/{t}/range/1/day/{f}/{t}"


def polygon_bars(ticker: str, api_key: str, from_d: date, to_d: date) -> pd.DataFrame:
    url = BASE.format(ticker, from_d.isoformat(), to_d.isoformat())
    r = requests.get(url, params={"apiKey": api_key, "adjusted": "true", "limit": 5000}, timeout=30)
    r.raise_for_status()
    res = r.json()
    rows = []
    for b in res.get("results", []):
        rows.append({
            "date": pd.Timestamp(b["t"], unit="ms").date(),
            "ticker": ticker,
            "open": b.get("o"), "high": b.get("h"), "low": b.get("l"),
            "close": b.get("c"), "volume": b.get("v"),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--tickers", default=None, help="comma list; default = monitored universe")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        print("No POLYGON_API_KEY set — skipping Polygon ingest.")
        print("Get a free key at https://polygon.io and export POLYGON_API_KEY, then re-run.")
        return

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = sorted(pd.read_parquet(STOCKS)["ticker"].astype(str).str.upper().unique()) if STOCKS.exists() else []

    to_d = date.today()
    from_d = to_d - timedelta(days=args.days * 2)  # buffer for weekends
    frames = []
    for t in tickers:
        try:
            df = polygon_bars(t, api_key, from_d, to_d)
            if len(df):
                frames.append(df)
                print(f"  {t}: {len(df)} bars")
        except Exception as e:
            print(f"  {t}: ERR {e}")
        time.sleep(0.21)  # free tier = 5 req/s
    if not frames:
        print("No bars fetched.")
        return
    new = pd.concat(frames, ignore_index=True)
    if args.save:
        existing = pd.read_parquet(PRICES) if PRICES.exists() else pd.DataFrame()
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(["date", "ticker"], keep="last")
        combined.to_parquet(PRICES, index=False)
        print(f"\nAppended {len(new)} polygon bars → {PRICES} ({len(combined)} total rows)")


if __name__ == "__main__":
    main()
