#!/usr/bin/env python3
"""
Fix fundamentals.parquet - corrected version
"""

import pandas as pd
from analytics_common import atomic_write_parquet
import numpy as np
from pathlib import Path

FUND_PATH = Path('fundamentals.parquet')
PRICES_PATH = Path('daily_prices/')

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

# Get latest yfinance shares for each ticker
latest = fund.sort_values('as_of_date').groupby('ticker').tail(1)
yf_shares = latest[latest['source'].isin(['yfinance', 'yfinance_history']) & latest['shares_outstanding'].notna()][['ticker', 'shares_outstanding']]
yf_shares_dict = dict(zip(yf_shares['ticker'], yf_shares['shares_outstanding']))
print(f"YFinance shares for {len(yf_shares_dict)} tickers")

# Fill EDGAR shares from yfinance
edgar_mask = fund['source'] == 'edgar'
missing_shares = edgar_mask & fund['shares_outstanding'].isna()
fund.loc[missing_shares, 'shares_outstanding'] = fund.loc[missing_shares, 'ticker'].map(yf_shares_dict)
filled = missing_shares & fund['shares_outstanding'].notna()
print(f"Filled {filled.sum()} EDGAR shares from yfinance")

# Compute price at date
fund['price_at_date'] = fund.apply(lambda r: get_price_at_date(r['ticker'], r['as_of_date']), axis=1)

# Market cap
mcap_mask = fund['shares_outstanding'].notna() & fund['price_at_date'].notna() & fund['market_cap'].isna()
fund.loc[mcap_mask, 'market_cap'] = fund.loc[mcap_mask, 'shares_outstanding'] * fund.loc[mcap_mask, 'price_at_date']
print(f"Computed market_cap for {mcap_mask.sum()} rows")

# PB ratio
pb_mask = fund['market_cap'].notna() & fund['shareholders_equity'].notna() & (fund['shareholders_equity'] > 0) & fund['pb_ratio'].isna()
fund.loc[pb_mask, 'pb_ratio'] = fund.loc[pb_mask, 'market_cap'] / fund.loc[pb_mask, 'shareholders_equity']
print(f"Computed PB ratio for {pb_mask.sum()} rows")

# Debt to equity
de_mask = fund['total_debt'].notna() & fund['shareholders_equity'].notna() & (fund['shareholders_equity'] > 0) & fund['debt_to_equity'].isna()
fund.loc[de_mask, 'debt_to_equity'] = fund.loc[de_mask, 'total_debt'] / fund.loc[de_mask, 'shareholders_equity']
print(f"Computed D/E for {de_mask.sum()} rows")

# ROE
roe_mask = fund['net_income_quarterly'].notna() & fund['shareholders_equity'].notna() & (fund['shareholders_equity'] > 0) & fund['roe'].isna()
fund.loc[roe_mask, 'roe'] = fund.loc[roe_mask, 'net_income_quarterly'] / fund.loc[roe_mask, 'shareholders_equity']
print(f"Computed ROE for {roe_mask.sum()} rows")

# ROIC - use EBIT * (1 - tax_rate) / invested_capital
# tax_rate = 1 - net_income/ebit if both positive, else 0.21
roic_mask = (
    fund['ebit'].notna() & 
    fund['total_debt'].notna() & 
    fund['shareholders_equity'].notna() & 
    fund['cash_and_equivalents'].notna() &
    fund['roic'].isna()
)
invested_capital = fund['total_debt'] + fund['shareholders_equity'] - fund['cash_and_equivalents']
# Effective tax rate
tax_rate = np.where(
    (fund['ebit'] > 0) & (fund['net_income_quarterly'].notna()) & (fund['net_income_quarterly'] < fund['ebit']),
    1 - fund['net_income_ttm'] / fund['ebit'],   # both TTM: ebit is a TTM figure
    0.21
)
tax_rate = np.clip(tax_rate, 0, 0.5)
nopat = fund['ebit'] * (1 - tax_rate)
fund.loc[roic_mask & (invested_capital > 0), 'roic'] = nopat[roic_mask & (invested_capital > 0)] / invested_capital[roic_mask & (invested_capital > 0)]
print(f"Computed ROIC for {(roic_mask & (invested_capital > 0)).sum()} rows")

# FCF margin = free_cash_flow / revenue_ttm -- FCF is TTM, so the denominator is
# TTM revenue.
fcfm_mask = fund['free_cash_flow'].notna() & fund['revenue_ttm'].notna() & (fund['revenue_ttm'] > 0) & fund['fcf_margin'].isna()
fund.loc[fcfm_mask, 'fcf_margin'] = fund.loc[fcfm_mask, 'free_cash_flow'] / fund.loc[fcfm_mask, 'revenue_ttm']
print(f"Computed FCF margin for {fcfm_mask.sum()} rows")

# Reinvestment rate = capex / NOPAT
rr_mask = fund['capital_expenditure_ttm'].notna() & fund['ebit'].notna() & (fund['ebit'] > 0) & fund['reinvestment_rate'].isna()
nopat_rr = fund['ebit'] * (1 - tax_rate)
fund.loc[rr_mask & (nopat_rr > 0), 'reinvestment_rate'] = fund.loc[rr_mask & (nopat_rr > 0), 'capital_expenditure_ttm'] / nopat_rr[rr_mask & (nopat_rr > 0)]
print(f"Computed reinvestment rate for {(rr_mask & (nopat_rr > 0)).sum()} rows")

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
print(f"Computed EV/EBITDA for {(ev_mask & (ebitda_approx > 0)).sum()} rows")

# market_cap_b
fund.loc[fund['market_cap'].notna() & fund['market_cap_b'].isna(), 'market_cap_b'] = fund['market_cap'] / 1e9

# total_assets_b
fund.loc[fund['total_assets'].notna() & fund['total_assets_b'].isna(), 'total_assets_b'] = fund['total_assets'] / 1e9

# mktcap_to_assets
mta_mask = fund['market_cap'].notna() & fund['total_assets'].notna() & (fund['total_assets'] > 0) & fund['mktcap_to_assets'].isna()
fund.loc[mta_mask, 'mktcap_to_assets'] = fund.loc[mta_mask, 'market_cap'] / fund.loc[mta_mask, 'total_assets']
print(f"Computed mktcap_to_assets for {mta_mask.sum()} rows")

# Drop helper column
fund = fund.drop(columns=['price_at_date'])

# Save
print("\nSaving updated fundamentals...")
atomic_write_parquet(fund, FUND_PATH)
print(f"Saved: {len(fund)} rows")

# Verify improvements
latest2 = fund.sort_values('as_of_date').groupby('ticker').tail(1)
print("\n=== IMPROVED COVERAGE (Latest Quarter) ===")
for col in ['roic', 'roe', 'debt_to_equity', 'fcf_margin', 'reinvestment_rate', 
            'ev_ebitda', 'pb_ratio', 'market_cap', 'shares_outstanding',
            'market_cap_b', 'total_assets_b', 'mktcap_to_assets']:
    if col in latest2.columns:
        cnt = latest2[col].notna().sum()
        print(f"  {col}: {cnt}/{len(latest2)} ({cnt/len(latest2)*100:.1f}%)")

edgar_latest = latest2[latest2['source'] == 'edgar']
print(f"\nEDGAR tickers: {len(edgar_latest)}")
for col in ['roic', 'roe', 'debt_to_equity', 'fcf_margin', 'reinvestment_rate', 
            'ev_ebitda', 'pb_ratio', 'market_cap', 'shares_outstanding']:
    if col in edgar_latest.columns:
        cnt = edgar_latest[col].notna().sum()
        print(f"  EDGAR {col}: {cnt}/{len(edgar_latest)} ({cnt/len(edgar_latest)*100:.1f}%)")