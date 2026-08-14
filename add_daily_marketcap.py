#!/usr/bin/env python3
"""
Add daily market cap to daily_prices.parquet using DIRECT shares outstanding.

Decision: calculate daily market cap = close × shares_outstanding
where shares_outstanding is carried forward from fundamentals (EDGAR XBRL
shares tags), NOT re-inverted from market_cap/close.

This avoids the price-noise amplification and unit-error propagation
that happened when we did: daily_market_cap = close × (market_cap / close).
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent


def main() -> None:
    print("Loading daily_prices...")
    # PRESERVE all existing columns (close, adj_close, volume, ...) — only
    # add/recompute market_cap. Reading a subset here is what dropped
    # adj_close + volume in an earlier version.
    prices = pd.read_parquet(DATA_DIR / "daily_prices.parquet")
    if "close" not in prices.columns:
        raise SystemExit("daily_prices.parquet has no close column")
    keep = [c for c in prices.columns if c != "market_cap"]
    prices = prices[keep].copy()
    prices = prices.sort_values(["ticker", "date"])
    print(f"  {len(prices):,} rows, {prices['ticker'].nunique()} tickers, cols={list(prices.columns)}")

    print("Loading fundamentals (shares_outstanding)...")
    fund = pd.read_parquet(DATA_DIR / "fundamentals.parquet")
    fund = fund.sort_values(["ticker", "as_of_date"])
    print(f"  {len(fund):,} rows, {fund['ticker'].nunique()} tickers")

    # Normalize date dtypes to a common type for the merge. daily_prices'
    # `date` is DATE-native (datetime.date objects, stored as date32[day]),
    # while fundamentals' as_of_date arrives as datetime64 (fresh EDGAR/yfinance).
    # Merging object-vs-datetime64 raises "cannot merge object and datetime64".
    # Convert as_of_date to datetime.date objects to match prices' date column.
    fund["as_of_date"] = pd.to_datetime(fund["as_of_date"], errors="coerce").dt.date

    # Check shares_outstanding column
    if "shares_outstanding" not in fund.columns:
        print("  WARNING: shares_outstanding column missing — falling back to market_cap/close")
        # Fallback (old logic)
        fund_dates = fund[["ticker", "as_of_date", "market_cap"]].copy()
        fund_dates = fund_dates.merge(
            prices[["ticker", "date", "close"]],
            left_on=["ticker", "as_of_date"],
            right_on=["ticker", "date"],
            how="left"
        )
        fund_dates["shares_out"] = fund_dates["market_cap"] / fund_dates["close"]
        fund_dates = fund_dates.dropna(subset=["shares_out"])
        # Filter absurd implied shares (same bounds as backfill_edgar)
        fund_dates = fund_dates[(fund_dates["shares_out"] >= 1e6) & (fund_dates["shares_out"] <= 2e11)]
        fund_dates = fund_dates[["ticker", "as_of_date", "shares_out"]].drop_duplicates(subset=["ticker", "as_of_date"])
        print(f"  Computed shares_out at {len(fund_dates)} fundamental dates (fallback)")
    else:
        # Prefer direct EDGAR/yfinance shares; where a fundamental date has
        # market_cap but no shares_outstanding, derive shares = mcap/close at
        # that date. This fills tickers whose shares_outstanding was never
        # recorded (e.g. foreign/ETF/cyber names) using the market_cap that IS
        # present, instead of dropping the row.
        fund_dates = fund[["ticker", "as_of_date", "shares_outstanding", "market_cap"]].copy()
        fund_dates = fund_dates.merge(
            prices[["ticker", "date", "close"]],
            left_on=["ticker", "as_of_date"], right_on=["ticker", "date"], how="left"
        )
        # derived shares = market_cap / close at that as_of date (only where
        # shares_outstanding is missing AND we have both mcap and close)
        derived = fund_dates["market_cap"] / fund_dates["close"]
        fund_dates["shares_out"] = fund_dates["shares_outstanding"].fillna(derived)
        fund_dates = fund_dates.dropna(subset=["shares_out"])
        # Sanity: real companies have 1M-200B shares
        fund_dates = fund_dates[(fund_dates["shares_out"] >= 1e6) & (fund_dates["shares_out"] <= 2e11)]
        fund_dates = fund_dates[["ticker", "as_of_date", "shares_out"]].drop_duplicates(subset=["ticker", "as_of_date"])
        print(f"  Using direct + mcap-derived shares at {len(fund_dates)} fundamental dates")

    # Forward-fill shares to all trading dates per ticker
    prices = prices.merge(fund_dates, left_on=["ticker", "date"], right_on=["ticker", "as_of_date"], how="left")
    prices = prices.drop(columns=["as_of_date"])

    # Forward fill shares_out per ticker
    prices["shares_out"] = prices.groupby("ticker")["shares_out"].ffill()
    print(f"  After ffill: {prices['shares_out'].notna().sum():,} non-null / {len(prices):,} total")

    # Compute daily market cap = close × shares_out
    prices["market_cap"] = prices["close"] * prices["shares_out"]

    # Drop shares_out helper column, keep market_cap
    prices = prices.drop(columns=["shares_out"])

    # Verify
    mc = prices["market_cap"].dropna()
    print(f"  Market cap stats:")
    print(f"    non-null: {mc.notna().sum():,} / {len(prices):,}")
    print(f"    min: ${mc.min():,.0f}")
    print(f"    max: ${mc.max():,.0f}")
    print(f"    median: ${mc.median():,.0f}")

    # Save
    print("Saving daily_prices.parquet...")
    prices.to_parquet(DATA_DIR / "daily_prices.parquet", index=False)
    print("Done!")


if __name__ == "__main__":
    main()