#!/usr/bin/env python3
"""fetch_amplifier_history.py — add subsector-basket amplifier price history.

Why: the focused subsector baskets (macro_sector_shock.py SECTORS table)
name non-S&P amplifiers (TSM, ASML, SCCO, AEM, BTI, ...) that are NOT in
daily_prices.parquet. This fetches their full OHLCV history from yfinance
and appends it to daily_prices.parquet (source='yfinance'), so the
subsector shock signals use the real amplifier names instead of silently
dropping them.

Design: mirrors update_prices.py conventions — merges on (date, ticker),
keeps existing rows, sorts, dedupes. Only appends; never overwrites.

The amplifier set is declared HERE (not in macro_sector_shock) so the
fetch is reproducible and the unavailable names are explicit.

Usage: python fetch_amplifier_history.py [--max-years 30]
"""
from __future__ import annotations

import argparse
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent
PRICES = DATA_DIR / "daily_prices.parquet"

# non-S&P amplifier tickers used by the subsector baskets. Names absent
# here are either already in daily_prices (S&P members) or delisted
# (verified 2026-08: MRO/HES/CMA/SUM/CHX read 'possibly delisted' on
# Yahoo — HES was acquired by CVX, CHX delisted 2023).
AMPLIFIERS = [
    # energy
    "FTI", "CLNE", "ENB", "PBA", "CEG", "VST",
    # metals & mining
    "TSM", "ASML", "SCCO", "TECK", "HBM", "VALE",
    "AEM", "KGC", "RGLD", "GOLD", "AGI",
    "CLF", "RS", "CMC", "WOR", "X",
    # chemicals / construction
    "OLN", "LYB", "EXP", "UVV", "OSK",
    # other
    "BTI", "CP", "CNI", "AAL", "SNOW", "DDOG", "PARA",
    "MRNA", "INCY", "WAL", "OWL", "BAM",
]


def load_prices():
    if PRICES.exists():
        return pd.read_parquet(PRICES)
    return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "source"])


def main(max_years: int = 30):
    import yfinance as yf

    existing = load_prices()
    have = set(existing["ticker"].astype(str).str.upper())
    todo = [t for t in AMPLIFIERS if t.upper() not in have]
    print(f"amplifiers: {len(AMPLIFIERS)} declared, {len(AMPLIFIERS) - len(todo)} already in data, "
          f"{len(todo)} to fetch")
    if not todo:
        return

    fetched = []
    start = datetime(date.today().year - max_years, 1, 1)
    for t in todo:
        try:
            df = yf.download(t, start=start, progress=False, auto_adjust=False)
            if df is None or df.empty:
                print(f"  {t}: no data (delisted?), skipped")
                continue
            # yfinance returns MultiIndex columns [('Close','TSM'), ...]
            # with a 'Ticker' level — flatten to the single price level.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df = df.rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume",
            })
            if "close" not in df.columns or df["close"].dropna().empty:
                print(f"  {t}: no close data, skipped")
                continue
            df["ticker"] = t.upper()
            df["source"] = "yfinance"
            df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
            # DATE-native: the sink is date32[day]; pandas reads that back
            # as python datetime.date objects. Convert to datetime.date
            # BEFORE concat so the column stays date32 (writing datetime64
            # into the object column corrupts the dtype — caught when
            # min() hit mixed types).
            df["date"] = pd.to_datetime(df["date"]).dt.date
            fetched.append(df)
            print(f"  {t}: {len(df)} rows {df['date'].min()} -> {df['date'].max()}")
        except Exception as e:
            print(f"  {t}: FAIL {str(e)[:60]}")

    if not fetched:
        print("nothing fetched")
        return
    new = pd.concat(fetched, ignore_index=True)
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).drop_duplicates(
        subset=["date", "ticker"], keep="last")
    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(table, PRICES)
    print(f"\nSaved {len(combined)} price rows to {PRICES} "
          f"(+{len(new)} new from {len(fetched)} amplifiers)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-years", type=int, default=30)
    args = ap.parse_args()
    main(max_years=args.max_years)
