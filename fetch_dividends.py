"""Fetch trailing dividend history (yfinance) for the coverage universe.

Feeds shareholder-yield (item 10): sy = trailing-12m dividends / close.
Writes dividends_cache.parquet (ticker, ex_date, amount) as TYPED parquet.

Default is gap-only: tickers already in the cache are skipped. The old
path read a 300-name temp coverage file and OVERWROTE the cache, which
is why T/HPQ/AES/HMC/BAYRY (real payers) had zero rows and income
screens silently dropped them.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "dividends_cache.parquet"
DELAY_S = 0.6


def _load_cache() -> pd.DataFrame:
    if not OUT.exists():
        return pd.DataFrame(columns=["ticker", "ex_date", "amount"])
    dc = pd.read_parquet(OUT)
    dc["ticker"] = dc["ticker"].astype(str).str.upper()
    dc["ex_date"] = pd.to_datetime(dc["ex_date"]).dt.date
    dc["amount"] = pd.to_numeric(dc["amount"], errors="coerce")
    return dc.dropna(subset=["ticker", "ex_date", "amount"])


def _universe(args) -> list[str]:
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.tickers_file:
        p = Path(args.tickers_file)
        if p.exists():
            return [t.strip().upper() for t in p.read_text().splitlines() if t.strip()]
    from analytics_common import liquid_listed_tickers
    return sorted(liquid_listed_tickers())


def _fetch_one(tk: str) -> list[dict]:
    d = yf.Ticker(tk).dividends
    if d is None or len(d) == 0:
        return []
    d = d.dropna()
    rows = []
    for dt, amt in d.items():
        ex = dt.date() if hasattr(dt, "date") else pd.Timestamp(dt).date()
        rows.append({"ticker": tk, "ex_date": ex, "amount": float(amt)})
    return rows


def _save(dc: pd.DataFrame) -> None:
    dc = dc.copy()
    dc["ticker"] = dc["ticker"].astype(str).str.upper()
    dc["ex_date"] = pd.to_datetime(dc["ex_date"]).dt.date
    dc["amount"] = pd.to_numeric(dc["amount"], errors="coerce")
    dc = dc.dropna(subset=["ticker", "ex_date", "amount"])
    dc = dc.drop_duplicates(["ticker", "ex_date"], keep="last")
    dc = dc.sort_values(["ticker", "ex_date"])
    # typed parquet — never CSV
    dc.to_parquet(OUT, index=False)
    print(f"Wrote {OUT.name}: {len(dc)} rows, {dc['ticker'].nunique()} names")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", help="comma-separated tickers (targeted backfill)")
    ap.add_argument("--tickers-file", default="",
                    help="optional file of tickers; default = liquid_listed_tickers()")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch tickers already in the cache (default: skip them)")
    args = ap.parse_args()

    cache = _load_cache()
    have = set(cache["ticker"].unique()) if len(cache) else set()
    tickers = _universe(args)
    if not args.refresh:
        tickers = [t for t in tickers if t not in have]
    print(f"{len(tickers)} to fetch (cache has {len(have)} names)", flush=True)
    if not tickers:
        print("nothing missing")
        return

    rows = []
    empty = 0
    for i, tk in enumerate(tickers, 1):
        try:
            got = _fetch_one(tk)
            if got:
                rows.extend(got)
            else:
                empty += 1
        except Exception as e:  # noqa: BLE001
            print(f"  skip {tk}: {e}", flush=True)
        if i % 10 == 0 or i == len(tickers):
            print(f"{i}/{len(tickers)} (+{len(rows)} rows, {empty} empty)", flush=True)
        time.sleep(DELAY_S)

    if not rows:
        print("no new dividend rows")
        return
    new = pd.DataFrame(rows)
    merged = pd.concat([cache, new], ignore_index=True)
    _save(merged)


if __name__ == "__main__":
    main()
