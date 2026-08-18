#!/usr/bin/env python3
"""
Comprehensive Data Audit — Prices, Fundamentals, and Post-Processed Metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

print("=" * 80)
print("COMPREHENSIVE DATA AUDIT")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 80)
print()

# ============================================================
# 1. DAILY PRICES AUDIT
# ============================================================
print("=" * 80)
print("1. DAILY PRICES AUDIT")
print("=" * 80)

prices = pd.read_parquet('daily_prices.parquet')
prices['date'] = pd.to_datetime(prices['date']).dt.date

print(f"\nTotal rows: {len(prices):,}")
print(f"Unique tickers: {prices['ticker'].nunique():,}")
print(f"Date range: {prices['date'].min()} → {prices['date'].max()}")
print()

# Check for completely missing columns
print("Missing values:")
for col in prices.columns:
    missing = prices[col].isna().sum()
    if missing > 0:
        print(f"  {col}: {missing:,} ({missing/len(prices)*100:.2f}%)")
print()

# Check for stale tickers (no recent prices)
recent_cutoff = pd.Timestamp.now() - pd.DateOffset(days=7)
recent_prices = prices[prices['date'] >= recent_cutoff.date()]
recent_tickers = recent_prices['ticker'].nunique()
stale_count = prices['ticker'].nunique() - recent_tickers
print(f"Tickers with prices in last 7 days: {recent_tickers:,}")
print(f"Stale tickers (no prices in last 7 days): {stale_count:,}")
print()

# Check for duplicates
dupes = prices.duplicated(subset=['ticker', 'date'], keep=False)
print(f"Duplicate (ticker, date) rows: {dupes.sum():,}")
print()

# Check price statistics
print(f"Adj close stats:")
print(f"  Min: {prices['adj_close'].min():.4f}")
print(f"  Max: {prices['adj_close'].max():.2f}")
print(f"  Mean: {prices['adj_close'].mean():.2f}")
print(f"  Zero prices: {(prices['adj_close'] == 0).sum():,}")
print(f"  Negative prices: {(prices['adj_close'] < 0).sum():,}")
print()

# Check market_cap coverage
print(f"Market cap stats:")
print(f"  Non-null: {prices['market_cap'].notna().sum():,}")
print(f"  Null: {prices['market_cap'].isna().sum():,}")
print(f"  Min: {prices['market_cap'].min():,.0f}")
print(f"  Max: {prices['market_cap'].max():,.0f}")
print()

# ============================================================
# 2. FUNDAMENTALS AUDIT
# ============================================================
print("=" * 80)
print("2. FUNDAMENTALS AUDIT")
print("=" * 80)

fund = pd.read_parquet('fundamentals.parquet')
fund['as_of_date'] = pd.to_datetime(fund['as_of_date']).dt.date

print(f"\nTotal rows: {len(fund):,}")
print(f"Unique tickers: {fund['ticker'].nunique():,}")
print(f"Date range: {fund['as_of_date'].min()} → {fund['as_of_date'].max()}")
print()

# Check for completely missing columns
print("Missing values (key columns):")
key_cols = ['net_income_quarterly', 'revenue_quarterly', 'free_cash_flow', 'operating_income_quarterly', 
            'market_cap', 'total_assets', 'shareholders_equity', 'total_debt',
            'roe', 'roic', 'debt_to_equity', 'pb_ratio', 'ev_ebitda', 'fcf_margin']
for col in key_cols:
    if col in fund.columns:
        missing = fund[col].isna().sum()
        print(f"  {col}: {missing:,} ({missing/len(fund)*100:.1f}%)")
    else:
        print(f"  {col}: COLUMN MISSING")
print()

# Check for stale tickers
recent_cutoff = pd.Timestamp.now() - pd.DateOffset(days=90)
recent_fund = fund[fund['as_of_date'] >= recent_cutoff.date()]
stale_count = fund['ticker'].nunique() - recent_fund['ticker'].nunique()
print(f"Tickers with fundamentals in last 90 days: {recent_fund['ticker'].nunique():,}")
print(f"Stale tickers (no fundamentals in last 90 days): {stale_count:,}")
print()

# Check for duplicates
dupes = fund.duplicated(subset=['ticker', 'as_of_date'], keep=False)
print(f"Duplicate (ticker, as_of_date) rows: {dupes.sum():,}")
print()

# Check for future dates
future = fund[fund['as_of_date'] > datetime.now().date()]
print(f"Rows with future dates: {len(future):,}")
print()

# Check source distribution
print("Source distribution:")
print(fund['source'].value_counts().to_string())
print()

# ============================================================
# 3. MONITORED STOCKS COVERAGE
# ============================================================
print("=" * 80)
print("3. MONITORED STOCKS COVERAGE")
print("=" * 80)

mon = pd.read_parquet('monitored_stocks.parquet')

print(f"\nTotal monitored: {len(mon):,}")
print(f"Active: {(mon['status'] == 'active').sum():,}")
print()

# Price coverage
mon_with_prices = set(mon['ticker'].unique()) & set(prices['ticker'].unique())
mon_without_prices = set(mon['ticker'].unique()) - set(prices['ticker'].unique())
print(f"Monitored with price data: {len(mon_with_prices):,}")
print(f"Monitored WITHOUT price data: {len(mon_without_prices):,}")
if mon_without_prices:
    print(f"  Missing: {sorted(mon_without_prices)[:20]}")
print()

# Fundamentals coverage
mon_with_fund = set(mon['ticker'].unique()) & set(fund['ticker'].unique())
mon_without_fund = set(mon['ticker'].unique()) - set(fund['ticker'].unique())
print(f"Monitored with fundamentals: {len(mon_with_fund):,}")
print(f"Monitored WITHOUT fundamentals: {len(mon_without_fund):,}")
if mon_without_fund:
    print(f"  Missing: {sorted(mon_without_fund)[:20]}")
print()

# Recent fundamentals coverage (last 90 days)
mon_with_recent_fund = set(mon['ticker'].unique()) & set(recent_fund['ticker'].unique())
mon_without_recent_fund = set(mon['ticker'].unique()) - set(recent_fund['ticker'].unique())
print(f"Monitored with recent fundamentals (90d): {len(mon_with_recent_fund):,}")
print(f"Monitored WITHOUT recent fundamentals: {len(mon_without_recent_fund):,}")
if mon_without_recent_fund:
    print(f"  Missing: {sorted(mon_without_recent_fund)[:20]}")
print()

# ============================================================
# 4. POST-PROCESSED METRICS TABLES
# ============================================================
print("=" * 80)
print("4. POST-PROCESSED METRICS TABLES")
print("=" * 80)

# Find all parquet files
parquet_files = []
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.parquet'):
            parquet_files.append(os.path.join(root, f))

parquet_files.sort()

print(f"\nFound {len(parquet_files)} parquet files:")
print()

# Key post-processed tables to check
key_tables = [
    'preferred_metrics_history.parquet',
    'fundamentals_history_backfill.parquet',
    'damodaran_quality_ranked.parquet',
    'sector_analysis_latest.parquet',
    'sector_summary.parquet',
    'signal_aggregator_results.parquet',
    'buy_candidates.parquet',
    'backtest_results.parquet',
]

for table in key_tables:
    if os.path.exists(table):
        try:
            df = pd.read_parquet(table)
            size_mb = os.path.getsize(table) / (1024 * 1024)
            print(f"{table}:")
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Size: {size_mb:.1f} MB")
            if 'as_of_date' in df.columns:
                print(f"  Date range: {df['as_of_date'].min()} → {df['as_of_date'].max()}")
            elif 'date' in df.columns:
                print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
            print(f"  Unique tickers: {df['ticker'].nunique() if 'ticker' in df.columns else 'N/A'}")
            print()
        except Exception as e:
            print(f"{table}: ERROR loading - {e}")
            print()
    else:
        print(f"{table}: NOT FOUND")
        print()

# ============================================================
# 5. STALE DATA SUMMARY
# ============================================================
print("=" * 80)
print("5. STALE DATA SUMMARY")
print("=" * 80)

print(f"""
CRITICAL ISSUES:
1. Daily Prices:
   - {stale_count:,} stale tickers (no prices in last 7 days)
   - {(prices['adj_close'] == 0).sum():,} zero-price rows
   - {prices['market_cap'].isna().sum():,} missing market caps

2. Fundamentals:
   - {stale_count:,} stale tickers (no fundamentals in last 90 days)
   - {len(future):,} rows with future dates
   - {(fund['net_income_quarterly'].isna()).sum():,} missing net_income values

3. Monitored Stocks:
   - {len(mon_without_prices):,} monitored without price data
   - {len(mon_without_fund):,} monitored without fundamentals
   - {len(mon_without_recent_fund):,} monitored without recent fundamentals (90d)

4. Post-Processed Tables:
   - preferred_metrics_history: {len(pd.read_parquet('preferred_metrics_history.parquet')):,} rows
   - Check for stale date ranges and missing monitored stocks

RECOMMENDATIONS:
1. Remove stale tickers from prices and fundamentals
2. Fix future dates in fundamentals
3. Backfill missing data for monitored stocks
4. Refresh post-processed metrics tables after data cleanup
5. Add data freshness checks to daily automation
""")