#!/usr/bin/env python3
"""
backfill_marketcap.py — Backfill missing market_cap values.

Computes market_cap = price × shares_outstanding
Sources shares from fundamentals.parquet or Yahoo Finance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices/"
FUND = DATA_DIR / "fundamentals.parquet"


def backfill_from_fundamentals():
    """
    Use shares_outstanding from fundamentals to compute market_cap.
    This is the preferred method when fundamentals data is available.
    """
    prices = pd.read_parquet(PRICES)
    prices['date'] = pd.to_datetime(prices['date']).dt.date
    
    fund = pd.read_parquet(FUND)
    fund['as_of_date'] = pd.to_datetime(fund['as_of_date']).dt.date
    
    # Get latest shares_outstanding for each ticker
    latest_shares = (
        fund.sort_values('as_of_date')
        .groupby('ticker')
        .apply(lambda x: x.iloc[-1] if len(x) > 0 else None)
        .reset_index(drop=True)
    )
    
    if latest_shares.empty:
        print("No shares_outstanding data in fundamentals")
        return
    
    shares_map = latest_shares.set_index('ticker')['shares_outstanding'].to_dict()
    
    # Compute market_cap where missing
    missing_mask = prices['market_cap'].isna()
    print(f"Missing market_cap: {missing_mask.sum():,} rows")
    
    # Map shares to prices
    prices['shares_outstanding'] = prices['ticker'].map(shares_map)
    
    # Compute market_cap = price × shares
    computable = missing_mask & prices['shares_outstanding'].notna()
    prices.loc[computable, 'market_cap'] = (
        prices.loc[computable, 'adj_close'] * prices.loc[computable, 'shares_outstanding']
    )
    
    # Drop helper column
    prices = prices.drop(columns=['shares_outstanding'])
    
    # Save
    prices.to_parquet(PRICES, index=False)
    
    still_missing = prices['market_cap'].isna().sum()
    print(f"After backfill: {still_missing:,} rows still missing market_cap")
    print(f"Computed {computable.sum():,} new market_cap values")


def backfill_from_yfinance(tickers: list[str] = None):
    """
    Fallback: use yfinance to get shares_outstanding.
    Only for tickers where fundamentals data is unavailable.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not available")
        return
    
    prices = pd.read_parquet(PRICES)
    prices['date'] = pd.to_datetime(prices['date']).dt.date
    
    # Find tickers with missing market_cap
    missing_tickers = prices[prices['market_cap'].isna()]['ticker'].unique()
    
    if tickers:
        missing_tickers = [t for t in missing_tickers if t in tickers]
    
    print(f"Fetching shares for {len(missing_tickers)} tickers from yfinance...")
    
    shares_map = {}
    for i, ticker in enumerate(missing_tickers):
        try:
            info = yf.Ticker(ticker).info
            shares = info.get('sharesOutstanding')
            if shares and shares > 0:
                shares_map[ticker] = shares
        except Exception:
            pass
        
        if (i + 1) % 100 == 0:
            print(f"  Fetched {i+1}/{len(missing_tickers)}")
    
    # Apply
    for ticker, shares in shares_map.items():
        mask = (prices['ticker'] == ticker) & (prices['market_cap'].isna())
        prices.loc[mask, 'market_cap'] = prices.loc[mask, 'adj_close'] * shares
    
    prices.to_parquet(PRICES, index=False)
    print(f"Backfilled {len(shares_map)} tickers from yfinance")


if __name__ == "__main__":
    print("Backfilling market_cap from fundamentals...")
    backfill_from_fundamentals()
    
    # Optional: backfill remaining from yfinance
    # backfill_from_yfinance()
