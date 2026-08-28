#!/usr/bin/env python3
"""
update_prices.py — Daily OHLCV ingest for daily_prices/ with primary/fallback feeds.

Primary: Polygon.io bulk grouped endpoint (one request = ALL US stocks for a day).
  - Requires POLYGON_API_KEY env var.
  - No per-ticker rate limit; fast incremental (only missing dates).
Fallback: yfinance (per-ticker, rate-limited to ~2000 req/hr).
  - Used when Polygon key is absent, or for tickers Polygon misses.

Incremental: only fetches (date, ticker) pairs not already in daily_prices/.

Usage:
  python update_prices.py fetch [--days 5] [--save]
  python update_prices.py fetch --source yfinance   # force fallback only
  python update_prices.py fetch --source polygon    # force primary only (fail if no key)
  python update_prices.py manual TICK open close [--date YYYY-MM-DD]
  python update_prices.py from-csv prices.csv
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
PRICES_DIR = DATA_DIR / "daily_prices/"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"


def load_prices():
    if PRICES_DIR.exists():
        return pd.read_parquet(PRICES_DIR)
    return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "source"])


def save_prices(df):
    if isinstance(df["ticker"].dtype, pd.CategoricalDtype):
        df = df.copy()
        df["ticker"] = df["ticker"].astype(str)
    df = df.copy()
    df["date"] = df["date"].map(lambda d: d.date() if isinstance(d, pd.Timestamp) else d)
    df = df.sort_values(["date", "ticker"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, PRICES_DIR)
    print(f"Saved {len(df)} price rows to {PRICES_DIR}")


def get_active_tickers():
    if not STOCKS_FILE.exists():
        return []
    df = pd.read_parquet(STOCKS_FILE)
    return df[df["status"].isin(["active", "monitored"])]["ticker"].tolist()


def existing_dates() -> set[date]:
    dates = set()
    if not PRICES_DIR.exists():
        return dates
    for year_dir in PRICES_DIR.iterdir():
        if not year_dir.is_dir() or not year_dir.name.startswith("year="):
            continue
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                continue
            for pq in month_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(pq, columns=["date"])
                    for d in df["date"].unique():
                        if isinstance(d, pd.Timestamp):
                            dates.add(d.date())
                        elif isinstance(d, date):
                            dates.add(d)
                except Exception:
                    pass
    return dates


def existing_ticker_dates() -> dict[str, set[date]]:
    """Return {ticker: set of dates already present}."""
    td = {}
    if not PRICES_DIR.exists():
        return td
    for year_dir in PRICES_DIR.iterdir():
        if not year_dir.is_dir() or not year_dir.name.startswith("year="):
            continue
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                continue
            for pq in month_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(pq, columns=["date", "ticker"])
                    for _, row in df.iterrows():
                        d = row["date"]
                        if isinstance(d, pd.Timestamp):
                            d = d.date()
                        td.setdefault(str(row["ticker"]), set()).add(d)
                except Exception:
                    pass
    return td


def drop_phantom_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "volume" not in df.columns:
        return df
    d = df.copy()
    vol = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    phantom = vol <= 0
    n = int(phantom.sum())
    if n:
        print(f"  drop_phantom_rows: dropping {n} volume==0 rows")
    return d.loc[~phantom].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Polygon (primary)
# ---------------------------------------------------------------------------

def polygon_bulk_day(day: date, api_key: str) -> pd.DataFrame:
    """Fetch ALL US stocks for a single trading day via Polygon bulk endpoint."""
    import requests
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{day.isoformat()}"
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


def fetch_polygon(missing_dates: list[date], api_key: str) -> pd.DataFrame:
    """Fetch missing dates from Polygon. Returns DataFrame of new rows."""
    frames = []
    for day in missing_dates:
        try:
            df = polygon_bulk_day(day, api_key)
            if len(df):
                vol = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0)
                df = df.loc[vol > 0]
            if len(df) < 100:
                print(f"  {day}: skip closed/thin ({len(df)} live bars)")
                continue
            frames.append(df)
            print(f"  {day}: {len(df)} tickers from Polygon")
        except Exception as e:
            print(f"  {day}: ERR {e}")
        time.sleep(0.5)  # free tier: 5 req/min
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# yfinance (fallback)
# ---------------------------------------------------------------------------

def fetch_yfinance(tickers: list[str], days: int, batch_size: int = 50, max_workers: int = 4) -> pd.DataFrame:
    """Fetch tickers from yfinance in concurrent batches with rate-limit retry."""
    import yfinance as yf
    all_raw = []
    all_adj = []
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    print(f"  {len(batches)} batches of {batch_size}, {max_workers} workers")

    def fetch_batch(batch, adjusted, attempt=1):
        try:
            return yf.download(batch, period=f"{days}d", group_by="ticker",
                             auto_adjust=adjusted, progress=False, threads=False)
        except Exception as e:
            if "Rate limit" in str(e) or "Too Many" in str(e):
                if attempt < 3:
                    time.sleep(5 * attempt)
                    return fetch_batch(batch, adjusted, attempt + 1)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for batch in batches:
            futures[pool.submit(fetch_batch, batch, False)] = (batch, "raw")
            futures[pool.submit(fetch_batch, batch, True)] = (batch, "adj")
        for fut in as_completed(futures):
            batch, kind = futures[fut]
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    if kind == "raw":
                        all_raw.append(df)
                    else:
                        all_adj.append(df)
            except Exception as e:
                print(f"  {kind} batch failed: {e}")

    if not all_raw:
        return pd.DataFrame()

    data_raw = pd.concat(all_raw, axis=1) if len(all_raw) > 1 else all_raw[0]
    data_adj = pd.concat(all_adj, axis=1) if len(all_adj) > 1 else all_adj[0]

    rows = []
    for t in tickers:
        if t not in data_raw.columns.get_level_values(0):
            continue
        sub_raw = data_raw[t].dropna(how="all")
        sub_adj = data_adj[t].dropna(how="all") if t in data_adj.columns.get_level_values(0) else pd.DataFrame()
        for idx, row in sub_raw.iterrows():
            adj_close = None
            if len(sub_adj) and idx in sub_adj.index:
                adj_row = sub_adj.loc[idx]
                if not adj_row.isna().all():
                    adj_close = float(adj_row["Close"])
            rows.append({
                "date": idx.date() if hasattr(idx, "date") else idx,
                "ticker": t,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "adj_close": adj_close,
                "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                "source": "yfinance",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main fetch
# ---------------------------------------------------------------------------

def cmd_fetch(args):
    api_key = os.environ.get("POLYGON_API_KEY", "")
    source = args.source if hasattr(args, 'source') and args.source else "auto"

    use_polygon = (source in ("auto", "polygon")) and api_key
    use_yfinance = source in ("auto", "yfinance")

    if source == "polygon" and not api_key:
        print("ERROR: --source polygon requires POLYGON_API_KEY")
        return

    all_tickers = get_active_tickers()
    if not all_tickers:
        print("No monitored stocks found.")
        return

    # Find missing dates
    have_dates = existing_dates()
    to_d = date.today() - timedelta(days=1)
    from_d = to_d - timedelta(days=args.days)
    missing_dates = []
    for i in range((to_d - from_d).days + 1):
        day = from_d + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        if day not in have_dates:
            missing_dates.append(day)

    print(f"Active universe: {len(all_tickers)} tickers")
    print(f"Existing dates: {len(have_dates)}")
    print(f"Missing dates: {len(missing_dates)} ({missing_dates[0] if missing_dates else 'none'} → {missing_dates[-1] if missing_dates else 'none'})")

    frames = []

    # Primary: Polygon (fetch ALL tickers for missing dates in 1 request/day)
    if use_polygon and missing_dates:
        print(f"\n=== Polygon (primary) ===")
        poly_df = fetch_polygon(missing_dates, api_key)
        if len(poly_df):
            poly_df = drop_phantom_rows(poly_df)
            frames.append(poly_df)
            print(f"Polygon: {len(poly_df)} rows")

    # Fallback: yfinance for tickers Polygon missed (or if no Polygon key)
    if use_yfinance:
        if use_polygon and missing_dates:
            # Find tickers Polygon didn't cover for missing dates
            poly_tickers = set(poly_df["ticker"].unique()) if len(poly_df) else set()
            yf_tickers = [t for t in all_tickers if t not in poly_tickers]
            print(f"\n=== yfinance (fallback) ===")
            print(f"Polygon covered {len(poly_tickers)} tickers")
            print(f"yfinance fallback for {len(yf_tickers)} tickers")
        else:
            yf_tickers = all_tickers
            print(f"\n=== yfinance (no Polygon key) ===")
            print(f"Fetching {len(yf_tickers)} tickers")

        if yf_tickers:
            yf_df = fetch_yfinance(yf_tickers, args.days)
            if len(yf_df):
                yf_df = drop_phantom_rows(yf_df)
                frames.append(yf_df)
                print(f"yfinance: {len(yf_df)} rows")

    if not frames:
        print("No new data fetched.")
        return

    new_df = pd.concat(frames, ignore_index=True)
    new_df["date"] = pd.to_datetime(new_df["date"])

    if args.save:
        existing = load_prices()
        combined = pd.concat([existing, new_df], ignore_index=True)
        save_prices(combined)
    else:
        print(f"\nFetched {len(new_df)} rows (dry run, --save to persist)")


def cmd_manual(args):
    existing = load_prices()
    d = pd.to_datetime(args.date) if args.date else pd.Timestamp.now().normalize()
    row = {
        "date": d,
        "ticker": args.ticker.upper(),
        "open": float(args.open),
        "high": max(float(args.open), float(args.close)) * 1.002,
        "low": min(float(args.open), float(args.close)) * 0.998,
        "close": float(args.close),
        "volume": 0,
        "source": "manual",
    }
    new_df = pd.DataFrame([row])
    combined = pd.concat([existing, new_df], ignore_index=True)
    save_prices(combined)
    print(f"Recorded {args.ticker.upper()} {d.date()} O={args.open} C={args.close}")


def cmd_from_csv(args):
    csv_df = read_csv(args.csv)
    csv_df["date"] = pd.to_datetime(csv_df["date"])
    csv_df["ticker"] = csv_df["ticker"].str.upper()
    if "high" not in csv_df.columns:
        csv_df["high"] = csv_df[["open", "close"]].max(axis=1)
    if "low" not in csv_df.columns:
        csv_df["low"] = csv_df[["open", "close"]].min(axis=1)
    if "volume" not in csv_df.columns:
        csv_df["volume"] = 0
    csv_df["source"] = "csv"
    existing = load_prices()
    combined = pd.concat([existing, csv_df], ignore_index=True)
    save_prices(combined)
    print(f"Imported {len(csv_df)} rows from {args.csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("fetch", help="Fetch daily prices (Polygon primary, yfinance fallback)")
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--save", action="store_true")
    p.add_argument("--source", choices=["auto", "polygon", "yfinance"], default="auto",
                   help="auto: Polygon if key, else yfinance. polygon: force primary. yfinance: force fallback.")

    p = sub.add_parser("manual")
    p.add_argument("ticker")
    p.add_argument("open", type=float)
    p.add_argument("close", type=float)
    p.add_argument("--date")

    p = sub.add_parser("from-csv")
    p.add_argument("csv")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return
    if args.cmd == "fetch":
        cmd_fetch(args)
    elif args.cmd == "manual":
        cmd_manual(args)
    elif args.cmd == "from-csv":
        cmd_from_csv(args)


if __name__ == "__main__":
    main()
