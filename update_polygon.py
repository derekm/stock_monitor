#!/usr/bin/env python3
"""
update_polygon.py — Daily OHLCV ingest from Polygon.io (production-grade
price feed), key-gated.

Why it exists: the architecture TODO "integrate data sources: Polygon
(production)". The repo ingests via yfinance (prototyping); Polygon is the
production alternative. This script pulls daily bars for the monitored
universe using the BULK grouped endpoint and appends them into
daily_prices.parquet (source='polygon').

Key-gated by design: requires POLYGON_API_KEY env var. Without it, the
script explains how to get a key and exits 0 (no crash in the automation).

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
BASE = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}"


def polygon_bulk_day(day: date, api_key: str) -> pd.DataFrame:
    """Fetch ALL US stocks for a single trading day via bulk endpoint."""
    url = BASE.format(date=day.isoformat())
    r = requests.get(url, params={"apiKey": api_key, "adjusted": "true", "limit": 50000}, timeout=60)
    if r.status_code == 429:
        print(f"  {day}: rate limited")
        return pd.DataFrame()
    r.raise_for_status()
    res = r.json()
    rows = []
    for b in res.get("results", []):
        ts = pd.Timestamp(b["t"], unit="ms", tz="UTC").tz_convert(None)
        rows.append({
            "date": ts,
            "ticker": b["T"],
            "open": b.get("o"), "high": b.get("h"), "low": b.get("l"),
            "close": b.get("c"), "volume": b.get("v"),
            "adj_close": b.get("c"),
            "source": "polygon",
            "market_cap": None,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=252*5, help="Days of history to pull (default: 5 years)")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        print("No POLYGON_API_KEY set — skipping Polygon ingest.")
        print("Get a free key at https://polygon.io and export POLYGON_API_KEY, then re-run.")
        return

    to_d = date.today() - timedelta(days=1)  # yesterday at most (today's data not ready)
    from_d = to_d - timedelta(days=args.days)
    frames = []
    for i in range((to_d - from_d).days + 1):
        day = from_d + timedelta(days=i)
        # skip weekends
        if day.weekday() >= 5:
            continue
        try:
            df = polygon_bulk_day(day, api_key)
            if len(df):
                vol = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0)
                df = df.loc[vol > 0]
            if len(df) < 100:
                print(f"  {day}: skip closed/thin ({len(df)} live bars)")
                continue
            frames.append(df)
            print(f"  {day}: {len(df)} tickers")
        except Exception as e:
            print(f"  {day}: ERR {e}")
        time.sleep(0.5)  # free tier: 5 req/min for grouped endpoint

    if not frames:
        print("No bars fetched.")
        return

    new = pd.concat(frames, ignore_index=True)
    if args.save:
        existing = pd.read_parquet(PRICES) if PRICES.exists() else pd.DataFrame()
        # ensure date column is datetime64[ms] on both sides
        existing["date"] = pd.to_datetime(existing["date"])
        new["date"] = pd.to_datetime(new["date"])
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(["date", "ticker"], keep="last")
        combined.to_parquet(PRICES, index=False)
        print(f"\nAppended {len(new)} polygon bars -> {PRICES} ({len(combined)} total rows)")


if __name__ == "__main__":
    main()