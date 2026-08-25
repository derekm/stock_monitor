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

OTC COMPLEMENT (fixed 2026-08-24): the old skip test was "open/high/low each
have >=1 non-null", which permanently skipped 5,693 of 5,695 OTC stocks —
`update_prices --fetch` hands them a handful of recent bars, so they looked
"done" while holding a median of 9 price rows (listed names: 1,799). That is
why the OTC tape had no history and Bogle PMI could only price 85 of 5,351 OTC
names. `has_ohlc` now requires `--min-rows` (default 252) COMPLETE-OHLC rows,
so thin tickers get refetched. Because the merge is strictly additive, re-runs
are safe and idempotent.

Usage:
  python backfill_ohlcv.py                        # full universe
  python backfill_ohlcv.py --only otc             # the PMI complement (5,695 names)
  python backfill_ohlcv.py --only otc --limit 20  # test
  python backfill_ohlcv.py --limit 20             # first 20 tickers (test)
  python backfill_ohlcv.py --force                # refetch even tickers with OHLC
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
# The listed tape (TMI's gate). Everything else is the OTC/gray complement (PMI).
LISTED_EXCHANGES = {"NMS", "NYQ", "NCM", "NGM", "ASE"}


def load_prices():
    return pd.read_parquet(PRICES) if PRICES.exists() else pd.DataFrame()


def save_prices(df: pd.DataFrame):
    """Write the price table without ever silently dropping existing rows.

    DO NOT normalize Timestamp -> date here. `daily_prices.parquet` is
    datetime64[ms] and carries 112,217 rows stamped 20:00 alongside the 00:00
    session rows; mapping to calendar dates and then de-duplicating on
    (date, ticker) collapsed 22,980 of them, so a run that touched 9 OTC
    tickers silently deleted ~8 rows each from 6,465 LISTED tickers (MSFT,
    MMM, COST...). The date key is preserved exactly as stored, and the
    de-dup is a no-op guard rather than a lossy normalization.
    """
    df = df.copy()
    if isinstance(df["ticker"].dtype, pd.CategoricalDtype):
        df["ticker"] = df["ticker"].astype(str)
    before = len(df)
    df = df.sort_values(["date", "ticker"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    if len(df) != before:
        print(f"  WARNING save_prices de-dup removed {before - len(df)} exact (date,ticker) duplicates")
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


def has_ohlc(price_df: pd.DataFrame, ticker: str, min_rows: int = 252) -> bool:
    """True only if the ticker already has ENOUGH OHLC history to skip.

    The old test was `open/high/low each have >=1 non-null`, which silently
    skipped every OTC name: `update_prices --fetch` gives them a handful of
    recent OHLC bars, so 5,693 of 5,695 OTC tickers looked "done" while holding
    a median of 9 rows. That is why the OTC tape was never backfilled and PMI
    could only price 85 names.

    Now a ticker is skipped only when it has at least `min_rows` rows carrying
    complete OHLC (default 252 = ~1 trading year). Thin tickers get refetched,
    and because the merge is strictly additive this is safe to re-run.
    """
    sub = price_df[price_df["ticker"] == ticker]
    if sub.empty:
        return False
    complete = sub[["open", "high", "low"]].notna().all(axis=1)
    return int(complete.sum()) >= min_rows


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


def merge_additive(prices: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """STRICTLY ADDITIVE merge of fetched OHLCV into the price table.

    Never overwrite existing close/volume/market_cap/adj_close. We only:
      1. FILL open/high/low where they are currently NaN in the existing rows,
      2. ADD brand-new (date, ticker) rows for dates the existing table lacks.
    Existing rows are preserved byte-for-byte in every other column.

    Extracted from main() so a long run can checkpoint mid-flight and stay
    resumable; the merge semantics are unchanged.

    Date keys are matched in the EXISTING table's dtype (datetime64[ms], which
    carries both 00:00 and 20:00 stamps). Coercing to datetime.date here would
    fail to match those rows and then collide two stamps onto one key — the bug
    that deleted 22,980 listed-ticker rows.
    """
    new_df = new_df.copy()
    if prices.empty:
        combined = new_df
        for c in SCHEMA_COLS:
            if c not in combined.columns:
                combined[c] = np.nan
        return combined[[c for c in SCHEMA_COLS if c in combined.columns]]

    prices = prices.copy()
    # Align the fetched date key to however the table already stores dates.
    if pd.api.types.is_datetime64_any_dtype(prices["date"]):
        new_df["date"] = pd.to_datetime(new_df["date"])
    else:
        new_df["date"] = new_df["date"].map(lambda d: d.date() if isinstance(d, pd.Timestamp) else d)

    if True:
        fetched = set(new_df["ticker"])
        idx_key = ["date", "ticker"]

        # 1) Fill OHLC into EXISTING rows where those columns are NaN, and
        #    refresh nothing else. Match on (date, ticker).
        fill_cols = ["open", "high", "low"]
        ex = prices[prices["ticker"].isin(fetched)].set_index(idx_key)
        ex = ex[~ex.index.duplicated(keep="last")]
        nd = new_df.set_index(idx_key)
        nd = nd[~nd.index.duplicated(keep="last")]
        overlap = ex.index.intersection(nd.index)
        if len(overlap):
            to_fill = ex.loc[overlap].copy()
            src = nd.loc[overlap]
            for c in fill_cols:
                missing = to_fill[c].isna()
                if missing.any():
                    to_fill.loc[missing, c] = src.loc[missing, c]
            keys = pd.MultiIndex.from_frame(prices[idx_key])
            prices = prices[~(prices["ticker"].isin(fetched) & keys.isin(overlap))]
            prices = pd.concat([prices, to_fill.reset_index()], ignore_index=True)

        # 2) Add brand-new (date, ticker) rows the table doesn't have.
        existing_keys = set(map(tuple, prices[idx_key].itertuples(index=False, name=None)))
        new_keys = list(map(tuple, new_df[idx_key].itertuples(index=False, name=None)))
        brand_new = new_df[[k not in existing_keys for k in new_keys]].copy()
        if len(brand_new):
            brand_new["source"] = "yfinance"
            # Carry forward market_cap from the nearest prior day of the same
            # ticker (never invent data; only re-align existing caps).
            # Vectorized: one sorted merge_asof over all fetched tickers at once.
            # The old form ran `prices[prices.ticker == t]` per fetched ticker,
            # i.e. 400 full scans of a 33M-row frame per checkpoint, which made a
            # single checkpoint merge take longer than the fetches it protected.
            if "market_cap" in prices.columns:
                caps = prices.loc[prices["ticker"].isin(fetched), ["date", "ticker", "market_cap"]].dropna(subset=["market_cap"])
                if len(caps):
                    caps = caps.copy()
                    caps["_d"] = pd.to_datetime(caps["date"]).astype("datetime64[ns]")
                    caps = caps.sort_values("_d")
                    bn = brand_new.copy()
                    bn["_row"] = np.arange(len(bn))
                    bn["_d"] = pd.to_datetime(bn["date"]).astype("datetime64[ns]")
                    bn = bn.sort_values("_d")
                    filled = pd.merge_asof(
                        bn[["_row", "_d", "ticker"]], caps[["_d", "ticker", "market_cap"]],
                        on="_d", by="ticker", direction="backward",
                    )
                    filled = filled.set_index("_row")["market_cap"].reindex(np.arange(len(brand_new)))
                    brand_new["market_cap"] = filled.values
            combined = pd.concat([prices, brand_new], ignore_index=True)
        else:
            combined = prices

    return combined[[c for c in SCHEMA_COLS if c in combined.columns]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max tickers (test)")
    ap.add_argument("--force", action="store_true", help="refetch even with OHLC")
    ap.add_argument("--delay", type=float, default=0.5, help="sleep between tickers")
    ap.add_argument("--min-rows", type=int, default=252,
                    help="skip a ticker only if it already has this many complete-OHLC rows (default 252)")
    ap.add_argument("--only", choices=["all", "otc", "listed"], default="all",
                    help="restrict to the OTC complement (PMI universe) or the listed tape")
    ap.add_argument("--checkpoint", type=int, default=250,
                    help="merge+save every N fetched tickers so a long run is resumable (0=only at end)")
    args = ap.parse_args()

    prices = load_prices()
    tickers = universe()

    if args.only != "all" and STOCKS.exists():
        m = pd.read_parquet(STOCKS, columns=["ticker", "instrument_type", "exchange"])
        m["ticker"] = m["ticker"].astype(str).str.upper()
        st = m[m["instrument_type"].eq("stock")]
        on_ex = st["exchange"].astype(str).isin(LISTED_EXCHANGES)
        want = set(st.loc[on_ex if args.only == "listed" else ~on_ex, "ticker"])
        tickers = [t for t in tickers if str(t).upper() in want]
        print(f"  --only {args.only}: {len(tickers)} tickers")

    if args.limit:
        tickers = tickers[: args.limit]

    # Precompute complete-OHLC row counts once instead of rescanning the whole
    # frame per ticker (that was O(tickers x rows) and dominated a 5.7k run).
    if prices.empty:
        ohlc_counts = {}
    else:
        complete = prices[["open", "high", "low"]].notna().all(axis=1)
        ohlc_counts = prices.loc[complete, "ticker"].astype(str).str.upper().value_counts().to_dict()

    print(f"Universe: {len(tickers)} tickers (OHLC coverage "
          f"{prices['open'].notna().mean() if 'open' in prices else 0:.1%})")

    new_frames = []
    n_skipped = 0
    for i, t in enumerate(tickers, 1):
        if not args.force and ohlc_counts.get(str(t).upper(), 0) >= args.min_rows:
            n_skipped += 1
            continue
        rows = fetch_ohlcv(t)
        if rows is not None and len(rows):
            new_frames.append(rows)
            print(f"  [{i}/{len(tickers)}] {t}: {len(rows)} rows")
        else:
            print(f"  [{i}/{len(tickers)}] {t}: no rows")
        time.sleep(args.delay + random.random() * 0.5)

        # Checkpoint: merge+save mid-run so a multi-hour OTC backfill is
        # resumable instead of losing everything on a timeout.
        if args.checkpoint and len(new_frames) >= args.checkpoint:
            print(f"  -- checkpoint at ticker {i}: merging {len(new_frames)} frames")
            prices = merge_additive(prices, pd.concat(new_frames, ignore_index=True))
            save_prices(prices)
            new_frames = []

    if n_skipped:
        print(f"  skipped {n_skipped} tickers already holding >={args.min_rows} complete-OHLC rows")

    if not new_frames:
        print("Nothing new to backfill.")
        return

    new_df = pd.concat(new_frames, ignore_index=True)
    print(f"Fetched {len(new_df)} OHLCV rows for {new_df['ticker'].nunique()} tickers")

    combined = merge_additive(prices, new_df)
    save_prices(combined)

    # verify
    after = load_prices()
    print(f"\nFinal OHLC coverage: {after['open'].notna().mean():.1%} "
          f"({after['ticker'].nunique()} tickers, {len(after)} rows)")
    print(f"Row count delta: {len(after) - len(prices) if 'prices' in locals() else len(after)} "
          f"(only brand-new dates added; existing rows never dropped)")


if __name__ == "__main__":
    main()