#!/usr/bin/env python3
"""
backfill_historical.py - Populate daily_prices/ with historical OHLCV data.

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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
PRICES_FILE = DATA_DIR / "daily_prices/"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"


def drop_phantom_rows(df: pd.DataFrame, gap_threshold: float = 0.30) -> pd.DataFrame:
    """Drop non-session rows. volume==0 is sufficient. Same as update_prices."""
    if df.empty or "volume" not in df.columns:
        return df
    d = df.copy()
    vol = pd.to_numeric(d["volume"], errors="coerce").fillna(0)
    phantom = vol <= 0
    n = int(phantom.sum())
    if n:
        print(f"  drop_phantom_rows: dropping {n} volume==0 rows")
    return d.loc[~phantom].reset_index(drop=True)


def load_prices() -> pd.DataFrame:
    if PRICES_FILE.exists():
        df = pd.read_parquet(PRICES_FILE)
        # `date` is stored as a DATE column -> read back as datetime.date.
        # Do NOT re-cast to Timestamp; keep it a plain date end-to-end.
        if "adj_close" not in df.columns:
            # backfill for pre-adj_close schema: assume close was raw
            df["adj_close"] = df["close"]
        return df
    return pd.DataFrame(
        columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
    )


def save_prices(df: pd.DataFrame) -> None:
    df = df.copy()
    # `date` must already be datetime.date (normalized at yfinance ingestion).
    # Keep it a date; do NOT cast to Timestamp.
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    # Explicit conflict reporting: a (date,ticker) with >1 distinct adj_close
    # means two divergent pulls collided (the old bug). Surface it instead of
    # silently keep="last".
    pre = len(df)
    coll = (
        df.groupby(["date", "ticker"])["adj_close"].nunique()
        .gt(1).sum()
    )
    if coll:
        print(f"  ⚠ {int(coll)} (date,ticker) pairs have CONFLICTING adj_close — "
              f"keeping mean to reconcile.")
        df = (
            df.groupby(["date", "ticker"], as_index=False)
            .agg({
                "adj_close": "mean",
                "close": "mean",
                "open": "mean",
                "high": "mean",
                "low": "mean",
                "volume": "mean",
                "source": "first",
            })
        )
    else:
        df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df = df.sort_values(["date", "ticker"])
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, PRICES_FILE)
    print(f"✓ Saved {len(df)} total rows → {PRICES_FILE} (from {pre} pre-dedup, "
          f"{int(coll)} conflicting pairs reconciled)")


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
    # auto_adjust=True makes yfinance return ADJUSTED Close (splits/divs
    # applied) and DROPS the raw Adj Close column; to keep both the adjusted
    # series (for training) and the raw close (for reference) we fetch with
    # auto_adjust=False and compute adj_close ourselves from the 'Adj Close'
    # column.
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
        # group_by="ticker" yields MultiIndex columns even for a single ticker
        if isinstance(raw.columns, pd.MultiIndex):
            sub = raw[t].dropna(subset=["Close"]) if t in raw.columns.get_level_values(0) else raw.dropna(subset=["Close"])
            for idx, row in sub.iterrows():
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
                        "adj_close": float(row["Adj Close"]) if pd.notna(row.get("Adj Close")) else float(row["Close"]),
                        "volume": int(row["Volume"]) if pd.notna(row.get("Volume")) else 0,
                        "source": "yfinance",
                    }
                )
        else:
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
                        "adj_close": float(row["Adj Close"]) if pd.notna(row.get("Adj Close")) else float(row["Close"]),
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
                        "adj_close": float(row["Adj Close"]) if pd.notna(row.get("Adj Close")) else float(row["Close"]),
                        "volume": int(row["Volume"]) if pd.notna(row.get("Volume")) else 0,
                        "source": "yfinance",
                    }
                )

    if not rows:
        print("No price rows retrieved. Check network / ticker symbols.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # rows already carry datetime.date from ingestion; keep as date.
    # yfinance emits non-trading days (holidays) with Volume=0 and a stale
    # close — drop them so daily returns don't get impossible 400-500% spikes.
    df = drop_phantom_rows(df)
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
    # rows already carry datetime.date; keep as date (no Timestamp).
    print(f"Generated {len(df)} synthetic rows for {len(tickers)} tickers over {days} days")
    return df


def import_csv(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"date", "ticker", "open", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")
    df["date"] = df["date"].apply(
        lambda s: datetime.strptime(str(s)[:10], "%Y-%m-%d").date())
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


def detect_adjclose_rescales(existing: pd.DataFrame, new_df: pd.DataFrame,
                              tol: float = 0.01) -> dict:
    """Flag tickers whose adj_close CHANGED on already-stored dates.

    yfinance re-resolves adj_close retroactively after a corporate action
    (split/dividend/special), so a ticker's ENTIRE history can shift on a
    given day. Any stored (date,ticker) whose new adj_close differs from the
    old by > `tol` (1% by default) means the historical series moved ->
    models trained on the old values are stale and need a FULL retrain.

    Returns {ticker: max_rel_shift} for every rescaled ticker.
    """
    if "adj_close" not in new_df.columns or existing.empty:
        return {}
    have = existing.dropna(subset=["adj_close"])
    if have.empty:
        return {}
    new = new_df.dropna(subset=["adj_close"])
    merged = new.merge(
        have[["date", "ticker", "adj_close"]].rename(columns={"adj_close": "adj_old"}),
        on=["date", "ticker"], how="inner",
    )
    if merged.empty:
        return {}
    old = merged["adj_old"].to_numpy(dtype=float)
    nw = merged["adj_close"].to_numpy(dtype=float)
    # guard against near-zero prices
    denom = np.where(np.abs(old) < 1e-6, np.abs(nw) + 1e-6, np.abs(old))
    rel = np.abs(nw - old) / denom
    merged = merged.assign(_rel=rel)
    rescaled = (
        merged.groupby("ticker")["_rel"].max()
        .loc[lambda s: s > tol]
    )
    return {tk: round(float(v), 4) for tk, v in rescaled.items()}


def merge_and_save(new_df: pd.DataFrame, overwrite: bool = False,
                   rescale_manifest: str = "rescaled_tickers.json") -> None:
    if new_df.empty:
        return
    existing = load_prices()

    # Normalize date to a single dtype so merges/concat don't choke:
    # parquet stores datetime64[ms]; fresh yfinance rows arrive as date objects.
    existing["date"] = pd.to_datetime(existing["date"])
    new_df["date"] = pd.to_datetime(new_df["date"])

    if overwrite:
        # Remove any overlapping (date, ticker) pairs from existing
        keys = new_df[["date", "ticker"]].apply(tuple, axis=1)
        existing_keys = existing[["date", "ticker"]].apply(tuple, axis=1)
        existing = existing[~existing_keys.isin(keys)]
        print(f"  Overwrite mode: removed {len(keys)} overlapping rows from existing data")

    # Detect corporate-action rescales BEFORE combining
    rescaled = detect_adjclose_rescales(existing, new_df)
    if rescaled:
        import json
        from pathlib import Path
        manifest = Path(__file__).parent / rescale_manifest
        try:
            prior = json.loads(manifest.read_text()) if manifest.exists() else {}
        except Exception:
            prior = {}
        prior.update({tk: {"max_rel_shift": v, "seen": str(pd.Timestamp.now().date())}
                      for tk, v in rescaled.items()})
        manifest.write_text(json.dumps(prior, indent=2))
        print(f"  ⚠ {len(rescaled)} ticker(s) had adj_close RESCALED (corporate action) -> "
              f"FULL retrain needed: {', '.join(sorted(rescaled))[:200]}")
        print(f"  Manifest written: {manifest}")

    combined = pd.concat([existing, new_df], ignore_index=True)
    save_prices(combined)

    # Summary
    print("\nBackfill summary by ticker (new rows):")
    print(new_df.groupby("ticker").size().to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Historical price backfill for daily_prices/",
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
