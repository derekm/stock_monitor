#!/usr/bin/env python3
"""
backfill_historical.py - Populate daily_prices.parquet with historical OHLCV data.

Primary method (requires network + yfinance):
  python backfill_historical.py --period 1y
  python backfill_historical.py --start 2025-01-01 --end 2026-07-28
  python backfill_historical.py --tickers CF,MOS,NTR --period 6mo

Offline / testing methods:
  python backfill_historical.py --synthetic --days 30          # generate random-walk data
  python backfill_historical.py --from-csv historical.csv      # bulk import

The script merges on (date, ticker), keeps the newest source on conflict,
and never deletes existing rows unless --overwrite is passed.

Fisher index notes:
  - Backfill must include volume (quantity). yfinance Volume maps to `volume`.
  - Synthetic backfill should generate positive volumes for index stability.
  - Rebuild: python fisher_index.py --universe all --save

TTM-ready notes:
  - Granite TTM benefits from 512+ daily points of clean OHLCV history.
  - Prefer --period 2y when network available so context window is full.
  - Synthetic backfill is for pipeline tests only; replace before decision use.
  - After backfill: python ttm_features.py --index portfolio --save
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"


def load_prices() -> pd.DataFrame:
    if PRICES_FILE.exists():
        df = pd.read_parquet(PRICES_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame(
        columns=["date", "ticker", "open", "high", "low", "close", "volume", "source"]
    )


def save_prices(df: pd.DataFrame) -> None:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).drop_duplicates(
        subset=["date", "ticker"], keep="last"
    )
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, PRICES_FILE)
    print(f"✓ Saved {len(df)} total rows → {PRICES_FILE}")


def get_tickers(explicit: str | None = None, status_filter: list | None = None) -> list[str]:
    if explicit:
        return [t.strip().upper() for t in explicit.split(",") if t.strip()]
    if not STOCKS_FILE.exists():
        print("No monitored_stocks.parquet found and no --tickers given.")
        sys.exit(1)
    stocks = pd.read_parquet(STOCKS_FILE)
    if status_filter:
        stocks = stocks[stocks["status"].isin(status_filter)]
    return stocks["ticker"].tolist()


def fetch_yfinance(tickers: list[str], start: str | None, end: str | None, period: str | None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed. Install with: pip install yfinance")
        print("Falling back is not automatic — use --synthetic or --from-csv instead.")
        sys.exit(1)

    print(f"Fetching {len(tickers)} tickers via yfinance …")
    kwargs = {"group_by": "ticker", "auto_adjust": False, "progress": True, "threads": True}
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end or datetime.now().strftime("%Y-%m-%d")

    raw = yf.download(tickers, **kwargs)

    rows = []
    if len(tickers) == 1:
        t = tickers[0]
        for idx, row in raw.iterrows():
            if pd.isna(row.get("Close")):
                continue
            rows.append(
                {
                    "date": idx.date() if hasattr(idx, "date") else idx,
                    "ticker": t,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]) if pd.notna(row.get("Volume")) else 0,
                    "source": "yfinance",
                }
            )
    else:
        # Multi-index columns: (ticker, field)
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                print(f"  ⚠ No data returned for {t}")
                continue
            sub = raw[t].dropna(subset=["Close"])
            for idx, row in sub.iterrows():
                rows.append(
                    {
                        "date": idx.date() if hasattr(idx, "date") else idx,
                        "ticker": t,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]) if pd.notna(row.get("Volume")) else 0,
                        "source": "yfinance",
                    }
                )

    if not rows:
        print("No price rows retrieved. Check network / ticker symbols.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Retrieved {len(df)} rows across {df['ticker'].nunique()} tickers")
    return df


def generate_synthetic(tickers: list[str], days: int, seed: int = 42) -> pd.DataFrame:
    """Create realistic-looking random-walk prices for offline testing."""
    import numpy as np

    rng = np.random.default_rng(seed)
    # Base prices roughly matching the screenshot levels
    base_prices = {
        "CF": 120.0, "MOS": 22.0, "NTR": 65.0, "ICL": 5.2, "IPI": 34.0,
        "UAN": 115.0, "SMG": 68.0, "CTVA": 85.0, "LXU": 11.5, "ASIX": 21.0,
        "ANDE": 78.0, "FMC": 11.5, "KHC": 26.0, "CAG": 15.0, "BAYRY": 13.0,
    }

    end = datetime.now().date()
    dates = [end - timedelta(days=i) for i in range(days)][::-1]

    rows = []
    for t in tickers:
        price = base_prices.get(t, 50.0)
        for d in dates:
            # modest daily volatility
            ret = rng.normal(0.0005, 0.018)
            open_p = price
            close_p = price * (1 + ret)
            high_p = max(open_p, close_p) * (1 + abs(rng.normal(0, 0.005)))
            low_p = min(open_p, close_p) * (1 - abs(rng.normal(0, 0.005)))
            vol = int(rng.integers(200_000, 3_000_000))
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "open": round(open_p, 2),
                    "high": round(high_p, 2),
                    "low": round(low_p, 2),
                    "close": round(close_p, 2),
                    "volume": vol,
                    "source": "synthetic",
                }
            )
            price = close_p

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Generated {len(df)} synthetic rows for {len(tickers)} tickers over {days} days")
    return df


def import_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "ticker", "open", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].str.upper()
    if "high" not in df.columns:
        df["high"] = df[["open", "close"]].max(axis=1)
    if "low" not in df.columns:
        df["low"] = df[["open", "close"]].min(axis=1)
    if "volume" not in df.columns:
        df["volume"] = 0
    if "source" not in df.columns:
        df["source"] = "csv"
    print(f"Loaded {len(df)} rows from {path}")
    return df


def merge_and_save(new_df: pd.DataFrame, overwrite: bool = False) -> None:
    if new_df.empty:
        return
    existing = load_prices()

    if overwrite:
        # Remove any overlapping (date, ticker) pairs from existing
        keys = new_df[["date", "ticker"]].apply(tuple, axis=1)
        existing_keys = existing[["date", "ticker"]].apply(tuple, axis=1)
        existing = existing[~existing_keys.isin(keys)]
        print(f"  Overwrite mode: removed {len(keys)} overlapping rows from existing data")

    combined = pd.concat([existing, new_df], ignore_index=True)
    save_prices(combined)

    # Summary
    print("\nBackfill summary by ticker (new rows):")
    print(new_df.groupby("ticker").size().to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Historical price backfill for daily_prices.parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tickers", help="Comma-separated list (default: all active+monitored)")
    parser.add_argument("--status", nargs="+", default=["active", "monitored"],
                        help="Status filter when pulling from monitored_stocks")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--period", help="yfinance period: 1mo,3mo,6mo,1y,2y,5y,ytd,max")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic random-walk data")
    parser.add_argument("--days", type=int, default=90, help="Days of synthetic history")
    parser.add_argument("--from-csv", help="Path to CSV with date,ticker,open,close,...")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing (date,ticker) rows instead of keeping them")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic data")

    args = parser.parse_args()

    tickers = get_tickers(args.tickers, args.status)
    print(f"Target tickers ({len(tickers)}): {', '.join(tickers)}")

    if args.from_csv:
        new_df = import_csv(args.from_csv)
        # optional filter to requested tickers
        if args.tickers:
            new_df = new_df[new_df["ticker"].isin(tickers)]
    elif args.synthetic:
        new_df = generate_synthetic(tickers, args.days, args.seed)
    else:
        # yfinance path
        if not args.period and not args.start:
            # sensible default
            args.period = "1y"
            print("No --period or --start given; defaulting to --period 1y")
        new_df = fetch_yfinance(tickers, args.start, args.end, args.period)

    if new_df is not None and not new_df.empty:
        merge_and_save(new_df, overwrite=args.overwrite)
    else:
        print("Nothing to write.")


if __name__ == "__main__":
    main()
