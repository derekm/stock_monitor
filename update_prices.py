#!/usr/bin/env python3
"""
update_prices.py - Append daily open/close (and OHLC) to daily_prices.parquet.

In environments with internet + yfinance:
  python update_prices.py --fetch

Otherwise, manual entry or CSV import:
  python update_prices.py --manual TICK open close [--date YYYY-MM-DD]
  python update_prices.py --from-csv prices.csv

The script always merges on (date, ticker) and avoids duplicates.

Quantity / Fisher index notes:
  - `volume` is treated as quantity (q) for Laspeyres/Paasche/Fisher indexes.
  - Prefer real volume on every update; zeros are carried-forward in fisher_index.py.
  - After updates: python fisher_index.py --universe portfolio --save

TTM-ready notes:
  - Prefer full OHLCV (open, high, low, close, volume) for Granite multivariate panels.
  - Business-day frequency; gaps are ffilled in ttm_features.build_panel.
  - After updates run: python forecast_granite.py forecast --index portfolio --from-first-trade
"""

import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, date
from pathlib import Path

DATA_DIR = Path(__file__).parent
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"

def load_prices():
    if PRICES_FILE.exists():
        return pd.read_parquet(PRICES_FILE)
    return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "source"])

def save_prices(df):
    # Parquet round-trips can leave 'ticker' as an unordered Categorical;
    # sorting on it then throws "values is not ordered". Normalize to str.
    if isinstance(df["ticker"].dtype, pd.CategoricalDtype):
        df = df.copy()
        df["ticker"] = df["ticker"].astype(str)
    # DATE-native: existing parquet rows are datetime.date while newly fetched
    # rows are pd.Timestamp — mixed types cannot be sorted (Timestamp vs
    # datetime.date comparison raises). Normalize to datetime.date (the
    # canonical daily date-key type) so the sort is homogeneous and the
    # parquet sink stays date32[day].
    df = df.copy()
    df["date"] = df["date"].map(lambda d: d.date() if isinstance(d, pd.Timestamp) else d)
    df = df.sort_values(["date", "ticker"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, PRICES_FILE)
    print(f"Saved {len(df)} price rows to {PRICES_FILE}")

def get_active_tickers():
    if not STOCKS_FILE.exists():
        return []
    df = pd.read_parquet(STOCKS_FILE)
    return df[df["status"].isin(["active", "monitored"])]["ticker"].tolist()


def drop_phantom_rows(df: pd.DataFrame, gap_threshold: float = 0.30) -> pd.DataFrame:
    """Drop non-session rows. volume==0 is sufficient (NYSE holiday / Sunday bars).

    yfinance and some Polygon holiday prints emit Volume=0 with a stale close.
    The old 30% neighbor-gap test let those through when close sat near the
    prior session (AMD Labor Day). gap_threshold is unused; kept so callers
    do not break.
    """
    if df.empty or "volume" not in df.columns:
        return df
    d = df.copy()
    vol = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    phantom = vol <= 0
    n = int(phantom.sum())
    if n:
        print(f"  drop_phantom_rows: dropping {n} volume==0 rows")
    return d.loc[~phantom].reset_index(drop=True)

def cmd_fetch(args):
    """Attempt to fetch via yfinance (requires network)."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Use --manual or --from-csv.")
        return

    tickers = get_active_tickers()
    if not tickers:
        print("No monitored stocks found.")
        return

    print(f"Fetching last {args.days} days for {len(tickers)} tickers...")
    # Fetch unadjusted (raw) prices
    data_raw = yf.download(tickers, period=f"{args.days}d", group_by="ticker", auto_adjust=False, progress=False)
    # Fetch adjusted prices (for returns calculations)
    data_adj = yf.download(tickers, period=f"{args.days}d", group_by="ticker", auto_adjust=True, progress=False)

    rows = []
    if len(tickers) == 1:
        # Single ticker shape
        t = tickers[0]
        for idx, row in data_raw.iterrows():
            # Get adjusted close for same date if available
            adj_close = None
            if t in data_adj.columns.get_level_values(0):
                adj_row = data_adj[t].loc[idx] if idx in data_adj[t].index else None
                if adj_row is not None and not adj_row.isna().all():
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
    else:
        for t in tickers:
            if t not in data_raw.columns.get_level_values(0):
                print(f"  Warning: no data for {t}")
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

    if not rows:
        print("No price data retrieved (network issue or delisted).")
        return

    new_df = pd.DataFrame(rows)
    new_df["date"] = pd.to_datetime(new_df["date"])
    # yfinance returns non-trading days (holidays) as rows with volume=0 and a
    # stale/garbage close (e.g. $47 for a $300 stock on Labor Day). Those rows
    # create impossible 400-500% daily returns. Drop them BEFORE the merge.
    new_df = drop_phantom_rows(new_df)
    existing = load_prices()
    combined = pd.concat([existing, new_df], ignore_index=True)
    save_prices(combined)
    print(f"Added/updated {len(rows)} rows ({len(new_df)} after phantom-drop).")

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
    """CSV must have columns: date,ticker,open,close[,high,low,volume]"""
    csv_df = pd.read_csv(args.csv)
    csv_df["date"] = pd.to_datetime(csv_df["date"])
    csv_df["ticker"] = csv_df["ticker"].str.upper()
    if "high" not in csv_df.columns:
        csv_df["high"] = csv_df[["open", "close"]].max(axis=1)
    if "low" not in csv_df.columns:
        csv_df["low"] = csv_df[["open", "close"]].min(axis=1)
    if "volume" not in csv_df.columns:
        csv_df["volume"] = 0
    if "source" not in csv_df.columns:
        csv_df["source"] = "csv"
    existing = load_prices()
    combined = pd.concat([existing, csv_df], ignore_index=True)
    save_prices(combined)

def cmd_show(args):
    df = load_prices()
    if args.ticker:
        df = df[df["ticker"] == args.ticker.upper()]
    if args.last:
        df = df.sort_values("date").groupby("ticker").tail(args.last)
    print(df.sort_values(["date", "ticker"]).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Update daily price parquet")
    sub = parser.add_subparsers(dest="cmd")

    p_fetch = sub.add_parser("fetch", help="Fetch via yfinance (needs network)")
    p_fetch.add_argument("--days", type=int, default=5)
    p_fetch.set_defaults(func=cmd_fetch)

    p_man = sub.add_parser("manual")
    p_man.add_argument("ticker")
    p_man.add_argument("open", type=float)
    p_man.add_argument("close", type=float)
    p_man.add_argument("--date", default=None)
    p_man.set_defaults(func=cmd_manual)

    p_csv = sub.add_parser("from-csv")
    p_csv.add_argument("csv")
    p_csv.set_defaults(func=cmd_from_csv)

    p_show = sub.add_parser("show")
    p_show.add_argument("--ticker")
    p_show.add_argument("--last", type=int, default=5)
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
