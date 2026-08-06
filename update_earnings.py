#!/usr/bin/env python3
"""
update_earnings.py - Maintain earnings calendar: upcoming + historical earnings.

Canonical table: earnings_calendar.parquet
  ticker, earnings_date (DATE), eps_estimate, reported_eps, surprise_pct, source, last_updated

Usage:
  python update_earnings.py show
  python update_earnings.py show --ticker AAPL
  python update_earnings.py fetch --days 8        # yfinance get_earnings_dates per ticker
  python update_earnings.py fetch --ticker AAPL   # single ticker refresh

Fetch loop is capped (--max-tickers, default 60) and per-ticker wrapped in
try/except so one bad symbol can't kill the batch. Existing rows are kept;
new rows appended (dedup on ticker+earnings_date).
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
EARN_FILE = DATA_DIR / "earnings_calendar.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"


def load() -> pd.DataFrame:
    cols = [
        "ticker", "earnings_date", "eps_estimate", "reported_eps",
        "surprise_pct", "source", "last_updated",
    ]
    if EARN_FILE.exists():
        return pd.read_parquet(EARN_FILE)
    return pd.DataFrame(columns=cols)


def save(df: pd.DataFrame) -> None:
    df = df.sort_values(["ticker", "earnings_date"]).drop_duplicates(
        subset=["ticker", "earnings_date"], keep="last"
    )
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, EARN_FILE)
    print(f"Saved {len(df)} earnings rows → {EARN_FILE}")


def _ticker_list(max_tickers: int | None) -> list[str]:
    stocks = pd.read_parquet(STOCKS_FILE) if STOCKS_FILE.exists() else pd.DataFrame()
    if stocks.empty or "ticker" not in stocks.columns:
        return []
    tickers = sorted(stocks["ticker"].astype(str).str.upper().unique().tolist())
    # ETFs/mutual funds don't report earnings; skip obvious ones to save API calls
    skip = {".", "^", "=", "-", ":"}
    tickers = [t for t in tickers if not any(s in t for s in skip)]
    if max_tickers:
        tickers = tickers[:max_tickers]
    return tickers


def cmd_fetch(args) -> None:
    import yfinance as yf

    if args.ticker:
        tickers = [t.upper() for t in args.ticker.split(",") if t.strip()]
    else:
        tickers = _ticker_list(args.max_tickers)

    today = date.today()
    df = load()
    new_rows: list[dict] = []
    ok = 0
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            ed = tk.get_earnings_dates(limit=max(8, args.days))
            if ed is None or len(ed) == 0:
                continue
            for ts, row in ed.iterrows():
                # ts is tz-aware timestamp → date
                d = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
                est = row.get("EPS Estimate")
                rep = row.get("Reported EPS")
                sur = row.get("Surprise(%)")
                new_rows.append({
                    "ticker": t,
                    "earnings_date": d,
                    "eps_estimate": None if pd.isna(est) else float(est),
                    "reported_eps": None if pd.isna(rep) else float(rep),
                    "surprise_pct": None if pd.isna(sur) else float(sur),
                    "source": "yfinance",
                    "last_updated": today,
                })
            ok += 1
        except Exception as e:  # noqa: BLE001 - one bad ticker must not kill the batch
            print(f"  !! {t}: {e}")
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save(df)
    print(f"Fetched {len(new_rows)} rows for {ok}/{len(tickers)} tickers")


def cmd_show(args) -> None:
    df = load()
    if args.ticker:
        df = df[df["ticker"] == args.ticker.upper()]
    if args.days:
        cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=args.days * 30)
        df = df[pd.to_datetime(df["earnings_date"]) >= cutoff]
    df = df.sort_values(["ticker", "earnings_date"])
    print(df.to_string(index=False))
    print(f"\n{len(df)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_show = sub.add_parser("show", help="Print earnings rows")
    p_show.add_argument("--ticker", default=None)
    p_show.add_argument("--days", type=int, default=None, help="Keep only last N*30 days")
    p_show.set_defaults(fn=cmd_show)
    p_fetch = sub.add_parser("fetch", help="Fetch earnings dates from yfinance")
    p_fetch.add_argument("--ticker", default=None, help="Comma-separated tickers (default: all monitored)")
    p_fetch.add_argument("--days", type=int, default=8, help="Quarters of history to fetch")
    p_fetch.add_argument("--max-tickers", type=int, default=60, help="Cap for batch fetch")
    p_fetch.set_defaults(fn=cmd_fetch)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
