#!/usr/bin/env python3
"""backfill_ohlcv.py — backfill full OHLCV history for the entire universe.

The stored `daily_prices.parquet` is close+volume only for almost all tickers
(OHLC coverage ~0.5%): the daily history came from a close-only source, and the
Polygon flat-files that carry true OHLC are blocked on this plan. This script
fills the gap with yfinance's full OHLCV history (open/high/low/close/volume),
which goes back decades per ticker.

For each ticker it:
  - fetches period='max' daily OHLCV via yfinance (auto_adjust=False so the raw
    Close/Open/High/Low/Volume are the as-traded values; Adj Close is stored as
    adj_close)
  - merges STRICTLY ADDITIVELY into daily_prices.parquet: it only FILLS
    open/high/low where they are currently NaN in existing rows and ADDS
    brand-new (date, ticker) rows the table lacks. It never overwrites existing
    close/volume/market_cap/adj_close — existing rows are preserved, so no
    better-quality data (e.g. EDGAR market cap) is ever lost.
  - is RESUME-SAFE: tickers that already have OHLC coverage are skipped, and
    partial batches re-run harmlessly (fill is idempotent; new-date adds dedupe).

Rate-limit friendly: sleeps a short jitter between tickers. Run in the
background for the full universe (586 tickers, a few minutes to ~20 min).

Usage:
  python backfill_ohlcv.py                 # full universe
  python backfill_ohlcv.py --limit 20      # first 20 tickers (test)
  python backfill_ohlcv.py --force         # refetch even tickers with OHLC
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent
PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
SCHEMA_COLS = ["date", "ticker", "adj_close", "close", "open", "high", "low",
               "volume", "source", "market_cap"]


def load_prices():
    return pd.read_parquet(PRICES) if PRICES.exists() else pd.DataFrame()


def save_prices(df: pd.DataFrame):
    df = df.copy()
    if isinstance(df["ticker"].dtype, pd.CategoricalDtype):
        df["ticker"] = df["ticker"].astype(str)
    # DATE-native: canonical date key is datetime.date; normalize timestamps.
    df["date"] = df["date"].map(lambda d: d.date() if isinstance(d, pd.Timestamp) else d)
    df = df.sort_values(["date", "ticker"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, PRICES)
    print(f"  saved {len(df)} rows -> {PRICES}")


def universe() -> list[str]:
    """All tickers to backfill: monitored stocks + any already in prices."""
    have = set()
    if STOCKS.exists():
        m = pd.read_parquet(STOCKS)
        if "ticker" in m.columns:
            have |= set(m["ticker"].astype(str))
    if PRICES.exists():
        have |= set(pd.read_parquet(PRICES, columns=["ticker"])["ticker"].unique())
    return sorted(have)


def has_ohlc(price_df: pd.DataFrame, ticker: str) -> bool:
    sub = price_df[price_df["ticker"] == ticker]
    return bool(sub["open"].notna().any() and sub["high"].notna().any()
                and sub["low"].notna().any())


def fetch_ohlcv(ticker: str) -> pd.DataFrame | None:
    """Full yfinance OHLCV history for one ticker -> clean long-format rows."""
    import yfinance as yf
    try:
        data = yf.download(ticker, period="max", interval="1d",
                           auto_adjust=False, progress=False)
    except Exception as e:  # noqa: BLE001
        print(f"  {ticker}: download error: {e}")
        return None
    if data is None or data.empty:
        print(f"  {ticker}: no data")
        return None
    # yfinance returns a multi-level column index; flatten
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    date_col = "Date" if "Date" in data.columns else data.columns[0]
    out = pd.DataFrame({
        "date": pd.to_datetime(data[date_col]),
        "ticker": ticker,
        "open": data["Open"].astype(float),
        "high": data["High"].astype(float),
        "low": data["Low"].astype(float),
        "close": data["Close"].astype(float),
        "volume": pd.to_numeric(data["Volume"], errors="coerce").fillna(0).astype(np.int64),
    })
    if "Adj Close" in data.columns:
        out["adj_close"] = pd.to_numeric(data["Adj Close"], errors="coerce")
    return out.dropna(subset=["open", "high", "low", "close"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max tickers (test)")
    ap.add_argument("--force", action="store_true", help="refetch even with OHLC")
    ap.add_argument("--delay", type=float, default=0.5, help="sleep between tickers")
    args = ap.parse_args()

    prices = load_prices()
    tickers = universe()
    if args.limit:
        tickers = tickers[: args.limit]
    print(f"Universe: {len(tickers)} tickers (OHLC coverage "
          f"{prices['open'].notna().mean() if 'open' in prices else 0:.1%})")

    new_frames = []
    for i, t in enumerate(tickers, 1):
        if not args.force and has_ohlc(prices, t):
            print(f"  [{i}/{len(tickers)}] {t}: already has OHLC, skip")
            continue
        rows = fetch_ohlcv(t)
        if rows is not None and len(rows):
            new_frames.append(rows)
            print(f"  [{i}/{len(tickers)}] {t}: {len(rows)} rows")
        else:
            print(f"  [{i}/{len(tickers)}] {t}: no rows")
        time.sleep(args.delay + random.random() * 0.5)

    if not new_frames:
        print("Nothing new to backfill.")
        return

    new_df = pd.concat(new_frames, ignore_index=True)
    print(f"Fetched {len(new_df)} OHLCV rows for {new_df['ticker'].nunique()} tickers")

    # ── STRICTLY ADDITIVE MERGE ────────────────────────────────────────────
    # Never overwrite existing close/volume/market_cap/adj_close. We only:
    #   1. FILL open/high/low where they are currently NaN in the existing rows,
    #   2. ADD brand-new (date, ticker) rows for dates the existing table lacks.
    # Existing rows are preserved byte-for-byte in every other column.
    new_df = new_df.copy()
    new_df["date"] = new_df["date"].map(lambda d: d.date() if isinstance(d, pd.Timestamp) else d)

    if prices.empty:
        combined = new_df
        for c in SCHEMA_COLS:
            if c not in combined.columns:
                combined[c] = np.nan
    else:
        prices = prices.copy()
        # normalize existing date to datetime.date for the merge keys
        prices["date"] = prices["date"].map(lambda d: d.date() if isinstance(d, pd.Timestamp) else d)

        fetched = set(new_df["ticker"])
        idx_key = ["date", "ticker"]

        # 1) Fill OHLC into EXISTING rows where those columns are NaN, and
        #    refresh nothing else. Match on (date, ticker).
        fill_cols = ["open", "high", "low"]
        # only rows we may fill: fetched tickers, with a matching fetched row
        ex = prices[prices["ticker"].isin(fetched)].set_index(idx_key)
        nd = new_df.set_index(idx_key)
        # align: keep only overlapping keys
        overlap = ex.index.intersection(nd.index)
        if len(overlap):
            to_fill = ex.loc[overlap].copy()
            src = nd.loc[overlap]
            for c in fill_cols:
                missing = to_fill[c].isna()
                if missing.any():
                    to_fill.loc[missing, c] = src.loc[missing, c]
            prices = prices[~prices["ticker"].isin(fetched) | ~prices.set_index(idx_key).index.isin(overlap)]
            prices = pd.concat([prices, to_fill.reset_index()], ignore_index=True)

        # 2) Add brand-new (date, ticker) rows the table doesn't have.
        existing_keys = set(prices.set_index(idx_key).index)
        new_keys = new_df.set_index(idx_key).index
        brand_new = new_df[~new_keys.isin(existing_keys)].copy()
        if len(brand_new):
            brand_new["source"] = "yfinance"
            # carry forward market_cap from the nearest prior day of the same
            # ticker if present (never invent data; only re-align existing caps)
            for t in fetched:
                if "market_cap" in prices.columns:
                    mc = prices[prices["ticker"] == t][["date", "market_cap"]].dropna()
                    if len(mc):
                        mc = mc.set_index("date").sort_index()
                        bn = brand_new[brand_new["ticker"] == t]
                        if len(bn):
                            caps = pd.Series(index=pd.to_datetime(bn["date"]),
                                             data=np.nan, dtype="float64")
                            filled = mc.reindex(index=caps.index, method="ffill")
                            brand_new.loc[brand_new["ticker"] == t, "market_cap"] = filled.values
            combined = pd.concat([prices, brand_new], ignore_index=True)
        else:
            combined = prices

    # reorder to schema
    combined = combined[[c for c in SCHEMA_COLS if c in combined.columns]]
    save_prices(combined)

    # verify
    after = load_prices()
    print(f"\nFinal OHLC coverage: {after['open'].notna().mean():.1%} "
          f"({after['ticker'].nunique()} tickers, {len(after)} rows)")
    print(f"Row count delta: {len(after) - len(prices) if 'prices' in locals() else len(after)} "
          f"(only brand-new dates added; existing rows never dropped)")


if __name__ == "__main__":
    main()