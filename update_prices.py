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
import uuid
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


def save_prices(df, whole_hive: bool = False):
    """Persist price rows into the partitioned hive.

    whole_hive=False (default, fetch path): df is ONLY new gap rows; already
    absent from the hive, so writing them as new partition files is true
    append — no dedupe against existing, no rewrite. whole_hive=True: df is a
    recombined full hive (cmd_manual/cmd_from_csv single-row or small edits
    where the row may overwrite an existing (date, ticker) pair); each
    month's old files are removed so the partition holds exactly df's rows.
    """
    if isinstance(df["ticker"].dtype, pd.CategoricalDtype):
        df = df.copy()
        df["ticker"] = df["ticker"].astype(str)
    df = df.copy()
    # Collapse to calendar DATE (date32[day] in parquet) — the daily bar has
    # no time component; Polygon's 20:00 UTC = 16:00 ET close is the same
    # calendar day, and yfinance emits midnight. A datetime64[ms] column here
    # re-introduces the 00:00/20:00 two-row split (migration 2026-09-03).
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    # Canonical-source dedupe: if both polygon and yfinance carry the same
    # (date, ticker), keep Polygon (primary, self-consistent adj_close); the
    # reverse fills only where Polygon has nothing.
    rank = df.get("source", pd.Series("", index=df.index)).fillna("").astype(str).map(
        lambda s: {"polygon": 2, "yfinance": 1}.get(s, 0))
    df["_r"] = rank
    df = df.sort_values(["date", "ticker", "_r"], kind="stable")
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df = df.drop(columns=["_r"])
    # Append-only partitioned write. pq.write_table(root_dir) treats the
    # directory as a FILE and fails WinError 5 ("open daily_prices denied") —
    # that exact error killed every real save in the 2026-09-03 DAG run.
    # Write each year/month group as its own partition file; readers glob
    # *.parquet under daily_prices/, so nothing is rewritten or deleted.
    dt = pd.to_datetime(df["date"])
    df["_year"] = dt.dt.year
    df["_month"] = dt.dt.month
    n_total = 0
    for (y, m), g in df.groupby(["_year", "_month"]):
        part = PRICES_DIR / f"year={y}" / f"month={m}"
        part.mkdir(parents=True, exist_ok=True)
        if whole_hive:
            for old in part.glob("*.parquet"):
                old.unlink()
        out = part / f"{uuid.uuid4().hex}-0.parquet"
        body = g.drop(columns=["_year", "_month"])
        # Explicit date32[day] at the sink: pandas/pyarrow can otherwise
        # infer an object column of python dates as TIMESTAMP or string.
        dates = pa.array([d if hasattr(d, "year") else None for d in body["date"]], type=pa.date32())
        cols = {c: pa.array(body[c], from_pandas=True) for c in body.columns if c != "date"}
        table = pa.Table.from_arrays([dates] + list(cols.values()), names=["date"] + list(cols.keys()))
        pq.write_table(table, out)
        n_total += len(g)
        print(f"  wrote {len(g)} rows -> {out.name} (year={y} month={m})")
    print(f"Saved {n_total} price rows to {PRICES_DIR}")


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


def _partial_dates(window: list[date], have_dates: set[date], n_universe: int) -> list[date]:
    """Dates present but with <50% of the universe's tickers (interrupted writes)."""
    if not window or n_universe <= 0 or not PRICES_DIR.exists():
        return []
    cols = ["date", "ticker"]
    df = pd.read_parquet(PRICES_DIR, columns=cols)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    per_day = df.groupby("date")["ticker"].count()
    out = []
    for d in window:
        if d in have_dates and d in per_day.index:
            if per_day.loc[d] < 0.5 * n_universe:
                out.append(d)
    return out


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
        # Daily bar -> calendar date. Polygon's timestamp (t) is the exchange
        # session time (20:00 UTC = 16:00 ET close); keep only the date, never
        # the time-of-day, so the hive stays date-native.
        ts = pd.Timestamp(b["t"], unit="ms", tz="UTC").date()
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

def existing_pairs_between(dates: list[date]) -> set[tuple[date, str]]:
    """(date, ticker) pairs already in the hive, restricted to the window.

    Reads ONLY the year/month partitions overlapping `dates` (columns
    date+ticker), not the whole hive — cheap enough to run every daily fetch."""
    pairs: set[tuple[date, str]] = set()
    if not PRICES_DIR.exists() or not dates:
        return pairs
    lo, hi = min(dates), max(dates)
    for year_dir in PRICES_DIR.iterdir():
        if not year_dir.is_dir() or not year_dir.name.startswith("year="):
            continue
        try:
            y = int(year_dir.name.split("=")[1])
        except ValueError:
            continue
        if y < lo.year or y > hi.year:
            continue
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                continue
            try:
                m = int(month_dir.name.split("=")[1])
            except ValueError:
                continue
            if (y, m) < (lo.year, lo.month) or (y, m) > (hi.year, hi.month):
                continue
            for pq in month_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(pq, columns=["date", "ticker"])
                    d = pd.to_datetime(df["date"]).dt.date
                    for dd, tt in zip(d, df["ticker"].astype(str)):
                        if lo <= dd <= hi:
                            pairs.add((dd, tt))
                except Exception:
                    pass
    return pairs


def cmd_fetch(args):
    # Polygon key must resolve even when the DAG runner's env lacks it:
    # same .env fallback chain as ticker_news._polygon_key() (the DAG parent
    # process has no exported key, so os.environ alone silently degrades to a
    # whole-universe yfinance crawl and times out).
    api_key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not api_key:
        for p in (DATA_DIR / ".env", DATA_DIR.parent / ".env"):
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("POLYGON_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
            if api_key:
                break
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
    window = []
    for i in range((to_d - from_d).days + 1):
        day = from_d + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        window.append(day)
        if day not in have_dates:
            missing_dates.append(day)

    # Partial dates: present but with a small fraction of the universe (interrupted
    # fetch). Refetch them so a truncated day is completed, not left as a gap.
    partial_dates = _partial_dates(window, have_dates, len(all_tickers))
    for d in partial_dates:
        if d not in missing_dates:
            missing_dates.append(d)
    missing_dates.sort()

    print(f"Active universe: {len(all_tickers)} tickers")
    print(f"Existing dates: {len(have_dates)}")
    print(f"Missing dates: {len(missing_dates)} ({missing_dates[0] if missing_dates else 'none'} → {missing_dates[-1] if missing_dates else 'none'})")

    frames = []

    # Strict gap set: (date, ticker) pairs absent from the hive, over the
    # missing/partial dates only. Every fetched row must land in this set —
    # no re-stamping of already-present pairs (append-only semantics; the
    # yfinance window fetch returns the full period, so filter it down).
    needed = set()
    if missing_dates:
        have_pairs = existing_pairs_between(window)
        for d in missing_dates:
            for t in all_tickers:
                if (d, t) not in have_pairs:
                    needed.add((d, t))
    print(f"Needed (date, ticker) pairs: {len(needed)}")

    # Primary: Polygon (fetch ALL tickers for missing dates in 1 request/day)
    if use_polygon and missing_dates:
        print(f"\n=== Polygon (primary) ===")
        poly_df = fetch_polygon(missing_dates, api_key)
        if len(poly_df):
            poly_df = drop_phantom_rows(poly_df)
            if needed:
                poly_df = poly_df[
                    poly_df.apply(
                        lambda r: (r["date"].date() if hasattr(r["date"], "date") else r["date"], str(r["ticker"])) in needed,
                        axis=1,
                    )
                ]
            frames.append(poly_df)
            print(f"Polygon: {len(poly_df)} rows")

    # Fallback: yfinance for pairs Polygon couldn't cover (or no Polygon key).
    # Guarded on missing_dates: when the hive is already current this crawl of
    # the FULL universe has nothing to fill and previously ran for >1h every
    # day (observed DAG timeout 2026-09-03 after wave-0 gating).
    if use_yfinance and missing_dates:
        if use_polygon:
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
                if needed:
                    yf_df = yf_df[
                        yf_df.apply(
                            lambda r: (r["date"].date() if hasattr(r["date"], "date") else r["date"], str(r["ticker"])) in needed,
                            axis=1,
                        )
                    ]
                frames.append(yf_df)
                print(f"yfinance: {len(yf_df)} rows")

    if not frames:
        print("No new data fetched.")
        return

    new_df = pd.concat(frames, ignore_index=True)
    new_df["date"] = pd.to_datetime(new_df["date"])

    if args.save:
        # Append-only: save ONLY the newly fetched gap rows, never the
        # recombined hive. Passing `existing + new_df` made save_prices
        # rewrite every month partition each run (777 new files, ~2.4M dup
        # rows on 2026-09-03) — the gap filter already guarantees new_df
        # shares no (date, ticker) with the hive, so there is nothing to
        # merge against.
        save_prices(new_df)
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
    save_prices(combined, whole_hive=True)
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
    save_prices(combined, whole_hive=True)
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
