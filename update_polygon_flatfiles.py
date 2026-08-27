#!/usr/bin/env python3
"""
update_polygon_flatfiles.py — Daily OHLCV ingest from Massive.com Flat Files
(S3). Key-gated, production-grade bulk price feed.

Why it exists: the architecture TODO "integrate data sources: Polygon
(production)". Massive's S3 flat-files endpoint is the bulk alternative to
per-ticker REST — ONE gzipped CSV per trading day covering ALL U.S. equities
in the `us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz` path.

Credentials (endpoint https://files.massive.com, bucket flatfiles) are read
from a gitignored `massive_credentials.json` so the secret never lands in
the repo. Without credentials the script prints setup and exits 0 (no crash
in the daily automation).

Usage:
    python update_polygon_flatfiles.py [--days 5] [--save]
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from analytics_common import DATA_DIR

PRICES = DATA_DIR / "daily_prices/"
CRED_FILE = DATA_DIR / "massive_credentials.json"
ENDPOINT = "https://files.massive.com"
BUCKET = "flatfiles"
PREFIX = "us_stocks_sip/day_aggs_v1"


def load_credentials() -> dict:
    if CRED_FILE.exists():
        cred = json.loads(CRED_FILE.read_text())
        if cred.get("access_key_id") and cred.get("secret_access_key"):
            return cred
    # fall back to env vars
    ak = os.environ.get("MASSIVE_ACCESS_KEY_ID", "")
    sk = os.environ.get("MASSIVE_SECRET_ACCESS_KEY", "")
    if ak and sk:
        return {"access_key_id": ak, "secret_access_key": sk}
    return {}


def day_csv_path(day: date) -> str:
    return f"{PREFIX}/{day.year:04d}/{day.month:02d}/{day.isoformat()}.csv.gz"


def fetch_day(day: date, cred: dict) -> pd.DataFrame:
    import boto3
    from botocore.config import Config
    session = boto3.Session(
        aws_access_key_id=cred["access_key_id"],
        aws_secret_access_key=cred["secret_access_key"],
    )
    s3 = session.client("s3", endpoint_url=ENDPOINT, config=Config(signature_version="s3v4"))
    key = day_csv_path(day)
    buf = io.BytesIO()
    try:
        s3.download_fileobj(BUCKET, key, buf)
    except Exception as e:
        # KeyError from botocore means the object doesn't exist (holiday/closed)
        if "404" in str(e) or "NoSuchKey" in str(e) or "does not exist" in str(e).lower():
            return pd.DataFrame()
        raise
    buf.seek(0)
    with gzip.open(buf, "rt") as fh:
        raw = fh.read()
    df = pd.read_csv(io.StringIO(raw))
    if df.empty:
        return df
    df = df.rename(columns={"Ticker": "ticker"})
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.Timestamp(day)
    df["source"] = "polygon_flat"
    df["market_cap"] = None
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    cred = load_credentials()
    if not cred:
        print("No Massive flat-files credentials found — skipping Polygon flat-file ingest.")
        print(f"Create {CRED_FILE} (gitignored) as:")
        print('  {"access_key_id": "...", "secret_access_key": "..."}')
        print("or set MASSIVE_ACCESS_KEY_ID / MASSIVE_SECRET_ACCESS_KEY env vars.")
        return 0

    to_d = date.today() - timedelta(days=1)  # yesterday at most (11am ET availability)
    from_d = to_d - timedelta(days=args.days * 2)
    frames = []
    for i in range((to_d - from_d).days + 1):
        day = from_d + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        try:
            df = fetch_day(day, cred)
            if len(df):
                frames.append(df)
                print(f"  {day}: {len(df)} rows")
            else:
                print(f"  {day}: no file (holiday/closed)")
        except Exception as e:
            print(f"  {day}: ERR {e}")

    if not frames:
        print("No bars fetched.")
        return 1

    new = pd.concat(frames, ignore_index=True)
    # keep only columns present in the spine
    cols = ["date", "ticker", "open", "high", "low", "close", "volume", "source", "market_cap"]
    for c in cols:
        if c not in new.columns:
            new[c] = None
    new = new[cols]

    if args.save:
        existing = pd.read_parquet(PRICES) if PRICES.exists() else pd.DataFrame()
        existing["date"] = pd.to_datetime(existing["date"])
        new["date"] = pd.to_datetime(new["date"])
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(["date", "ticker"], keep="last")
        combined.to_parquet(PRICES, index=False)
        print(f"\nAppended {len(new)} flat-file bars -> {PRICES} ({len(combined)} total rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())