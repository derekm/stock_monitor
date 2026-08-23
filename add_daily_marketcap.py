#!/usr/bin/env python3
"""PIT daily market cap: adj_close × last known shares_outstanding.

Writes daily_mcap.parquet (date, ticker, shares, market_cap). Does not open
daily_prices for write unless --write-prices.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
OUT_PANEL = DATA_DIR / "daily_mcap.parquet"
SHARES_MIN = 5e5
SHARES_MAX = 2e11


def _snap(src: Path) -> Path:
    dest = Path(tempfile.gettempdir()) / f"mcap_{src.name}"
    if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dest)
    return dest


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def build_panel(years: int | None, stock_only: bool) -> pd.DataFrame:
    prices = pd.read_parquet(
        _snap(DATA_DIR / "daily_prices.parquet"),
        columns=["date", "ticker", "adj_close", "close"],
    )
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = _to_date(prices["date"])
    prices = prices.dropna(subset=["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    px = prices["adj_close"].where(prices["adj_close"].notna(), prices["close"])
    prices = prices.assign(px=px).dropna(subset=["px"])

    if stock_only:
        ms = DATA_DIR / "monitored_stocks.parquet"
        if ms.exists():
            keep = pd.read_parquet(ms, columns=["ticker", "instrument_type"])
            keep["ticker"] = keep["ticker"].astype(str).str.upper()
            stocks = set(keep.loc[keep["instrument_type"].eq("stock"), "ticker"])
            prices = prices[prices["ticker"].isin(stocks)]

    if years:
        cutoff = max(d for d in prices["date"] if d is not None) - pd.Timedelta(days=int(years * 365.25))
        cutoff = cutoff.date() if hasattr(cutoff, "date") else cutoff
        prices = prices[prices["date"] >= cutoff]

    fund = pd.read_parquet(
        _snap(DATA_DIR / "fundamentals.parquet"),
        columns=["ticker", "as_of_date", "shares_outstanding"],
    )
    fund["ticker"] = fund["ticker"].astype(str).str.upper()
    fund["as_of_date"] = _to_date(fund["as_of_date"])
    sh = fund.dropna(subset=["ticker", "as_of_date", "shares_outstanding"])
    sh = sh[(sh["shares_outstanding"] >= SHARES_MIN) & (sh["shares_outstanding"] <= SHARES_MAX)]
    sh = sh.rename(columns={"as_of_date": "date", "shares_outstanding": "shares"})
    sh = sh.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")

    prices["date"] = pd.to_datetime(prices["date"])
    sh["date"] = pd.to_datetime(sh["date"])
    prices = prices.sort_values("date")
    sh = sh.sort_values("date")
    prices = pd.merge_asof(prices, sh, on="date", by="ticker", direction="backward")
    prices["date"] = prices["date"].dt.date
    prices["market_cap"] = prices["px"] * prices["shares"]
    prices.loc[prices["market_cap"] <= 0, "market_cap"] = np.nan
    out = prices[["date", "ticker", "shares", "market_cap"]].dropna(subset=["market_cap"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--stock-only", action="store_true", default=True)
    ap.add_argument("--all-types", action="store_true", help="Include warrants/ADR/OTC")
    ap.add_argument("--write-prices", action="store_true",
                    help="Also write market_cap onto daily_prices (holds the live file)")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    stock_only = not args.all_types
    print(f"PIT mcap panel years={args.years} stock_only={stock_only}")
    panel = build_panel(args.years, stock_only)
    last = panel[panel["date"] == panel["date"].max()]
    print(f"  {len(panel):,} rows  {panel['ticker'].nunique()} names  last {panel['date'].max()} n={len(last):,}")
    print(f"  last median ${last['market_cap'].median():,.0f}  p99 ${last['market_cap'].quantile(0.99):,.0f}")
    if args.save:
        panel.to_parquet(OUT_PANEL, index=False)
        print(f"Saved {OUT_PANEL}")
    if args.write_prices:
        print("WARNING: --write-prices opens live daily_prices; skip if a writer is running")


if __name__ == "__main__":
    main()
