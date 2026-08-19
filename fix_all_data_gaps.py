#!/usr/bin/env python3
"""
Comprehensive data gap fix for fundamentals.parquet
1. Compute all derived metrics from raw EDGAR/yfinance data
2. Cross-fill BRK-A/BRK-B/BRK.B data
3. Use price data to compute market_cap where missing
4. Ensure no redundant retrievals going forward
"""

import pandas as pd
import numpy as np
from pathlib import Path

FUND_PATH = Path('fundamentals.parquet')
PRICES_PATH = Path('daily_prices.parquet')

print("Loading fundamentals...")
fund = pd.read_parquet(FUND_PATH)
fund['as_of_date'] = pd.to_datetime(fund['as_of_date']).dt.date
print(f"Loaded: {len(fund)} rows, {fund['ticker'].nunique()} tickers")

print("Loading prices...")
prices = pd.read_parquet(PRICES_PATH)
prices['date'] = pd.to_datetime(prices['date']).dt.date
print(f"Loaded: {len(prices)} rows, {prices['ticker'].nunique()} tickers")

# Build price lookup
price_lookup = {}
for ticker, grp in prices.groupby('ticker'):
    price_lookup[ticker] = dict(zip(grp['date'], grp['adj_close']))

def get_price_at_date(ticker, target_date):
    if ticker not in price_lookup:
        return np.nan
    dates = sorted(price_lookup[ticker].keys())
    for d in reversed(dates):
        if d <= target_date:
            return price_lookup[ticker][d]
    return np.nan

# ============================================================
# STEP 1: Cross-fill BRK-A / BRK-B / BRK.B (same company)
# ============================================================
print("\n=== Cross-filling BRK data ===")

# Get BRK.B yfinance data (has recent shares, market cap)
brk_dot = fund[fund['ticker'] == 'BRK.B'].sort_values('as_of_date')
brk_a_dates = set(fund[fund['ticker'] == 'BRK-A']['as_of_date'])
brk_b_dates = set(fund[fund['ticker'] == 'BRK-B']['as_of_date'])

# Company-level columns (same for A and B shares)
company_cols = ['revenue_quarterly', 'net_income_quarterly', 'free_cash_flow', 'capital_expenditure_ttm',
                'shareholders_equity', 'total_debt', 'cash_and_equivalents', 
                'total_assets', 'total_liabilities', 'ebit', 'operating_income_quarterly',
                'ebitda', 'roe', 'roic', 'debt_to_equity', 'fcf_margin', 
                'reinvestment_rate', 'interest_coverage', 'ev_ebitda', 'pb_ratio',
                'mktcap_to_assets', 'market_cap', 'market_cap_b', 'total_assets_b']

# Fill BRK-A and BRK-B from BRK.B yfinance data
for _, row in brk_dot.iterrows():
    date = row['as_of_date']
    for col in company_cols:
        if col in fund.columns and pd.notna(row.get(col)):
            for ticker in ['BRK-A', 'BRK-B']:
                mask = (fund['ticker'] == ticker) & (fund['as_of_date'] == date)
                if mask.any() and pd.isna(fund.loc[mask, col].values[0]):
                    fund.loc[mask, col] = row[col]

# Shares: BRK.B shares = BRK-B shares, BRK-A shares = BRK.B / 1500
for _, row in brk_dot.iterrows():
    date = row['as_of_date']
    if pd.notna(row.get('shares_outstanding')):
        mask_b = (fund['ticker'] == 'BRK-B') & (fund['as_of_date'] == date)
        if mask_b.any() and pd.isna(fund.loc[mask_b, 'shares_outstanding'].values[0]):
            fund.loc[mask_b, 'shares_outstanding'] = row['shares_outstanding']
        mask_a = (fund['ticker'] == 'BRK-A') & (fund['as_of_date'] == date)
        if mask_a.any() and pd.isna(fund.loc[mask_a, 'shares_outstanding'].values[0]):
            fund.loc[mask_a, 'shares_outstanding'] = row['shares_outstanding'] / 1500

# Also cross-fill between BRK-A and BRK-B for EDGAR data
brk_a = fund[fund['ticker'] == 'BRK-A'].set_index('as_of_date')
brk_b = fund[fund['ticker'] == 'BRK-B'].set_index('as_of_date')
common_dates = brk_a.index.intersection(brk_b.index)

for date in common_dates:
    a_row = brk_a.loc[date]
    b_row = brk_b.loc[date]
    for col in company_cols:
        if col in fund.columns:
            a_val = a_row.get(col, np.nan)
            b_val = b_row.get(col, np.nan)
            if pd.isna(a_val) and pd.notna(b_val):
                fund.loc[(fund['ticker'] == 'BRK-A') & (fund['as_of_date'] == date), col] = b_val
            elif pd.isna(b_val) and pd.notna(a_val):
                fund.loc[(fund['ticker'] == 'BRK-B') & (fund['as_of_date'] == date), col] = a_val

# ============================================================
# STEP 2: Fill shares for all tickers from yfinance latest
# ============================================================
print("=== Filling shares from yfinance ===")
latest = fund.sort_values('as_of_date').groupby('ticker').tail(1)
yf_shares = latest[latest['source'].isin(['yfinance', 'yfinance_history']) & latest['shares_outstanding'].notna()][['ticker', 'shares_outstanding']]
yf_shares_dict = dict(zip(yf_shares['ticker'], yf_shares['shares_outstanding']))
print(f"YFinance shares for {len(yf_shares_dict)} tickers")

# Fill missing shares for all tickers (use latest yfinance shares as approximation)
missing_shares_mask = fund['shares_outstanding'].isna()
fund.loc[missing_shares_mask, 'shares_outstanding'] = fund.loc[missing_shares_mask, 'ticker'].map(yf_shares_dict)
filled = missing_shares_mask & fund['shares_outstanding'].notna()
print(f"Filled {filled.sum()} shares from yfinance latest")

# ============================================================
# STEP 3: Compute price at date and market_cap
# ============================================================
print("=== Computing market_cap from price × shares ===")
fund['price_at_date'] = fund.apply(lambda r: get_price_at_date(r['ticker'], r['as_of_date']), axis=1)

mcap_mask = fund['shares_outstanding'].notna() & fund['price_at_date'].notna() & fund['market_cap'].isna()
fund.loc[mcap_mask, 'market_cap'] = fund.loc[mcap_mask, 'shares_outstanding'] * fund.loc[mcap_mask, 'price_at_date']
print(f"Computed market_cap for {mcap_mask.sum()} rows")

# ============================================================
# STEP 4: Compute all derived metrics
# ============================================================
print("=== Computing derived metrics ===")

# PB ratio
pb_mask = fund['market_cap'].notna() & fund['shareholders_equity'].notna() & (fund['shareholders_equity'] > 0) & fund['pb_ratio'].isna()
fund.loc[pb_mask, 'pb_ratio'] = fund.loc[pb_mask, 'market_cap'] / fund.loc[pb_mask, 'shareholders_equity']
print(f"PB ratio: {pb_mask.sum()}")

# Debt to equity
de_mask = fund['total_debt'].notna() & fund['shareholders_equity'].notna() & (fund['shareholders_equity'] > 0) & fund['debt_to_equity'].isna()
fund.loc[de_mask, 'debt_to_equity'] = fund.loc[de_mask, 'total_debt'] / fund.loc[de_mask, 'shareholders_equity']
print(f"D/E: {de_mask.sum()}")

# ROE
roe_mask = fund['net_income_quarterly'].notna() & fund['shareholders_equity'].notna() & (fund['shareholders_equity'] > 0) & fund['roe'].isna()
fund.loc[roe_mask, 'roe'] = fund.loc[roe_mask, 'net_income_quarterly'] / fund.loc[roe_mask, 'shareholders_equity']
print(f"ROE: {roe_mask.sum()}")

# ROIC
roic_mask = (
    fund['ebit'].notna() & 
    fund['total_debt'].notna() & 
    fund['shareholders_equity'].notna() & 
    fund['cash_and_equivalents'].notna() &
    fund['roic'].isna()
)
invested_capital = fund['total_debt'] + fund['shareholders_equity'] - fund['cash_and_equivalents']
tax_rate = np.where(
    (fund['ebit'] > 0) & fund['net_income_quarterly'].notna() & (fund['net_income_quarterly'] < fund['ebit']),
    1 - fund['net_income_ttm'] / fund['ebit'],   # both TTM: ebit is a TTM figure
    0.21
)
tax_rate = np.clip(tax_rate, 0, 0.5)
nopat = fund['ebit'] * (1 - tax_rate)
fund.loc[roic_mask & (invested_capital > 0), 'roic'] = nopat[roic_mask & (invested_capital > 0)] / invested_capital[roic_mask & (invested_capital > 0)]
print(f"ROIC: {(roic_mask & (invested_capital > 0)).sum()}")

# FCF margin
fcfm_mask = fund['free_cash_flow'].notna() & fund['revenue_quarterly'].notna() & (fund['revenue_quarterly'] > 0) & fund['fcf_margin'].isna()
fund.loc[fcfm_mask, 'fcf_margin'] = fund.loc[fcfm_mask, 'free_cash_flow'] / fund.loc[fcfm_mask, 'revenue_ttm']
print(f"FCF margin: {fcfm_mask.sum()}")

# Reinvestment rate
rr_mask = fund['capital_expenditure_ttm'].notna() & fund['ebit'].notna() & (fund['ebit'] > 0) & fund['reinvestment_rate'].isna()
nopat_rr = fund['ebit'] * (1 - tax_rate)
fund.loc[rr_mask & (nopat_rr > 0), 'reinvestment_rate'] = fund.loc[rr_mask & (nopat_rr > 0), 'capital_expenditure_ttm'] / nopat_rr[rr_mask & (nopat_rr > 0)]
print(f"Reinvestment rate: {(rr_mask & (nopat_rr > 0)).sum()}")

# EV/EBITDA
ev_mask = (
    fund['market_cap'].notna() & 
    fund['total_debt'].notna() & 
    fund['cash_and_equivalents'].notna() & 
    fund['ebit'].notna() & 
    fund['capital_expenditure_ttm'].notna() &
    fund['ev_ebitda'].isna()
)
ev = fund['market_cap'] + fund['total_debt'] - fund['cash_and_equivalents']
ebitda_approx = fund['ebit'] + fund['capital_expenditure_ttm'].abs()
fund.loc[ev_mask & (ebitda_approx > 0), 'ev_ebitda'] = ev[ev_mask & (ebitda_approx > 0)] / ebitda_approx[ev_mask & (ebitda_approx > 0)]
print(f"EV/EBITDA: {(ev_mask & (ebitda_approx > 0)).sum()}")

# Billions columns
fund.loc[fund['market_cap'].notna() & fund['market_cap_b'].isna(), 'market_cap_b'] = fund['market_cap'] / 1e9
fund.loc[fund['total_assets'].notna() & fund['total_assets_b'].isna(), 'total_assets_b'] = fund['total_assets'] / 1e9

# Market cap to assets
mta_mask = fund['market_cap'].notna() & fund['total_assets'].notna() & (fund['total_assets'] > 0) & fund['mktcap_to_assets'].isna()
fund.loc[mta_mask, 'mktcap_to_assets'] = fund.loc[mta_mask, 'market_cap'] / fund.loc[mta_mask, 'total_assets']
print(f"Mktcap/Assets: {mta_mask.sum()}")

# Drop helper
fund = fund.drop(columns=['price_at_date'])

# ============================================================
# STEP 5: Deduplicate and sort
# ============================================================
print("\n=== Deduplicating ===")
fund = fund.sort_values(['ticker', 'as_of_date']).drop_duplicates(
    subset=['ticker', 'as_of_date'], keep='first'
)

# Save
print("\nSaving...")
fund.to_parquet(FUND_PATH, index=False)
print(f"Saved: {len(fund)} rows, {fund['ticker'].nunique()} tickers")

# ============================================================
# VERIFICATION
# ============================================================
latest2 = fund.sort_values('as_of_date').groupby('ticker').tail(1)
print("\n=== COVERAGE SUMMARY (Latest Quarter) ===")
for col in ['roic', 'roe', 'debt_to_equity', 'fcf_margin', 'reinvestment_rate', 
            'ev_ebitda', 'pb_ratio', 'market_cap', 'shares_outstanding',
            'market_cap_b', 'total_assets_b', 'mktcap_to_assets', 'revenue_quarterly',
            'net_income_quarterly', 'free_cash_flow', 'ebit', 'total_debt', 'cash_and_equivalents']:
    if col in latest2.columns:
        cnt = latest2[col].notna().sum()
        print(f"  {col}: {cnt}/{len(latest2)} ({cnt/len(latest2)*100:.1f}%)")

# By source
print("\n=== BY SOURCE ===")
for src in ['edgar', 'yfinance', 'yfinance_history']:
    src_latest = latest2[latest2['source'] == src]
    if len(src_latest) > 0:
        print(f"  {src}: {len(src_latest)} tickers")
        for col in ['roic', 'roe', 'fcf_margin', 'market_cap', 'shares_outstanding']:
            if col in src_latest.columns:
                cnt = src_latest[col].notna().sum()
                print(f"    {col}: {cnt}/{len(src_latest)} ({cnt/len(src_latest)*100:.1f}%)")

# BRK check
print("\n=== BRK VERIFICATION ===")
for t in ['BRK-A', 'BRK-B']:
    row = latest2[latest2['ticker'] == t]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"  {t}: shares={r.get('shares_outstanding', 'NaN'):,.0f}, mcap={r.get('market_cap', 'NaN'):,.0f}, roic={r.get('roic', 'NaN')}, roe={r.get('roe', 'NaN'):.4f}, fcfm={r.get('fcf_margin', 'NaN'):.4f}, de={r.get('debt_to_equity', 'NaN'):.4f}")