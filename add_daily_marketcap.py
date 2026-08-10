#!/usr/bin/env python3
"""
Add daily market cap to daily_prices.parquet.

Decision: calculate daily market cap = close * shares_outstanding
where shares_outstanding is carried forward from fundamentals (market_cap_b / close at fundamental dates).

Adds column `market_cap` to daily_prices (absolute dollars).
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent

def main():
    print("Loading daily_prices...")
    prices = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["date", "ticker", "close"])
    prices = prices.sort_values(["ticker", "date"])
    print(f"  {len(prices):,} rows, {prices['ticker'].nunique()} tickers")

    print("Loading fundamentals...")
    fund = pd.read_parquet(DATA_DIR / "fundamentals.parquet")
    fund = fund.sort_values(["ticker", "as_of_date"])
    print(f"  {len(fund):,} rows, {fund['ticker'].nunique()} tickers")

    # Compute shares_outstanding at fundamental dates: market_cap_b (billions) * 1e9 / close
    # Need close price at fundamental as_of_date
    fund_dates = fund[["ticker", "as_of_date", "market_cap_b"]].copy()
    fund_dates = fund_dates.merge(
        prices[["ticker", "date", "close"]],
        left_on=["ticker", "as_of_date"],
        right_on=["ticker", "date"],
        how="left"
    )
    fund_dates["shares_out"] = fund_dates["market_cap_b"] * 1e9 / fund_dates["close"]

    # Detect outliers: market_cap_b should be in billions; reject > 100,000 (unrealistic for market cap in billions)
    fund_dates = fund_dates[fund_dates["market_cap_b"] <= 100000]
    fund_dates = fund_dates.dropna(subset=["shares_out"])
    fund_dates = fund_dates[["ticker", "as_of_date", "shares_out"]].drop_duplicates(subset=["ticker", "as_of_date"])
    print(f"  Computed shares_out at {len(fund_dates)} fundamental dates")

    # Forward-fill shares_out to all trading dates per ticker
    # For each ticker, merge fund_dates into prices and forward-fill
    prices = prices.merge(fund_dates, left_on=["ticker", "date"], right_on=["ticker", "as_of_date"], how="left")
    prices = prices.drop(columns=["as_of_date"])

    # Forward fill shares_out per ticker
    prices["shares_out"] = prices.groupby("ticker")["shares_out"].ffill()
    print(f"  After ffill: {prices['shares_out'].notna().sum():,} non-null / {len(prices):,} total")

    # Compute daily market cap
    prices["market_cap"] = prices["close"] * prices["shares_out"]

    # Drop shares_out helper column, keep market_cap
    prices = prices.drop(columns=["shares_out"])

    # Verify
    print(f"  Market cap stats:")
    mc = prices["market_cap"].dropna()
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