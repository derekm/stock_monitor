#!/usr/bin/env python3
"""
Fix BRK-A/BRK-B data gaps and ensure consistent data
"""

import pandas as pd
import numpy as np
from pathlib import Path

FUND_PATH = Path('fundamentals.parquet')

print("Loading fundamentals...")
fund = pd.read_parquet(FUND_PATH)
fund['as_of_date'] = pd.to_datetime(fund['as_of_date']).dt.date

# Get BRK-A and BRK-B data
brk_a = fund[fund['ticker'] == 'BRK-A'].sort_values('as_of_date').copy()
brk_b = fund[fund['ticker'] == 'BRK-B'].sort_values('as_of_date').copy()

print(f"BRK-A rows: {len(brk_a)}, BRK-B rows: {len(brk_b)}")

# BRK-A has complete data for all 75 quarters
# BRK-B has gaps in recent quarters
# We should copy BRK-A data to BRK-B, adjusting shares

# Key: 1 BRK-A = 1500 BRK-B shares
# So BRK-B shares_outstanding = BRK-A shares * 1500
# Revenue, net_income, FCF, assets, etc. should be the SAME (company-level)
# But per-share metrics differ

# Find quarters where BRK-B is missing but BRK-A has data
brk_a_dates = set(brk_a['as_of_date'])
brk_b_dates = set(brk_b['as_of_date'])

missing_in_b = brk_a_dates - brk_b_dates
print(f"Quarters in A but not B: {len(missing_in_b)}")

# For each missing quarter, create BRK-B row from BRK-A
new_rows = []
for date in missing_in_b:
    a_row = brk_a[brk_a['as_of_date'] == date].iloc[0]
    new_row = a_row.copy()
    new_row['ticker'] = 'BRK-B'
    # Convert shares: BRK-B shares = BRK-A shares * 1500
    if pd.notna(a_row.get('shares_outstanding')):
        new_row['shares_outstanding'] = a_row['shares_outstanding'] * 1500
    # Market cap, assets, revenue, etc. stay the same (company level)
    new_rows.append(new_row)

if new_rows:
    new_df = pd.DataFrame(new_rows)
    fund = pd.concat([fund, new_df], ignore_index=True)
    print(f"Added {len(new_rows)} BRK-B rows from BRK-A")

# Also fix existing BRK-B rows that have NaN but BRK-A has data
# Update BRK-B rows where source is NaN or data is missing
for date in brk_a_dates & brk_b_dates:
    a_row = brk_a[brk_a['as_of_date'] == date].iloc[0]
    b_mask = (fund['ticker'] == 'BRK-B') & (fund['as_of_date'] == date)
    b_rows = fund[b_mask]
    if len(b_rows) > 0:
        idx = b_rows.index[0]
        # If BRK-B has NaN revenue but BRK-A has it, copy over
        for col in ['revenue_quarterly', 'net_income_quarterly', 'free_cash_flow', 'capital_expenditure_ttm',
                    'shareholders_equity', 'total_debt', 'cash_and_equivalents', 'total_assets',
                    'total_liabilities', 'ebit', 'operating_income_quarterly', 'roe', 'roic', 
                    'debt_to_equity', 'fcf_margin', 'reinvestment_rate', 'market_cap',
                    'market_cap_b', 'total_assets_b', 'mktcap_to_assets']:
            if col in fund.columns:
                if pd.isna(fund.loc[idx, col]) and pd.notna(a_row.get(col)):
                    fund.loc[idx, col] = a_row[col]
        # Fix shares
        if pd.isna(fund.loc[idx, 'shares_outstanding']) and pd.notna(a_row.get('shares_outstanding')):
            fund.loc[idx, 'shares_outstanding'] = a_row['shares_outstanding'] * 1500

# Now recompute derived metrics for BRK-B
prices = pd.read_parquet('daily_prices.parquet')
prices['date'] = pd.to_datetime(prices['date']).dt.date

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

# Update BRK-B price and market cap
brk_b_mask = fund['ticker'] == 'BRK-B'
fund.loc[brk_b_mask, 'price_at_date'] = fund.loc[brk_b_mask].apply(
    lambda r: get_price_at_date(r['ticker'], r['as_of_date']), axis=1)

# Market cap
mcap_mask = brk_b_mask & fund['shares_outstanding'].notna() & fund['price_at_date'].notna() & fund['market_cap'].isna()
fund.loc[mcap_mask, 'market_cap'] = fund.loc[mcap_mask, 'shares_outstanding'] * fund.loc[mcap_mask, 'price_at_date']

# PB ratio
pb_mask = brk_b_mask & fund['market_cap'].notna() & fund['shareholders_equity'].notna() & (fund['shareholders_equity'] > 0) & fund['pb_ratio'].isna()
fund.loc[pb_mask, 'pb_ratio'] = fund.loc[pb_mask, 'market_cap'] / fund.loc[pb_mask, 'shareholders_equity']

# D/E
de_mask = brk_b_mask & fund['total_debt'].notna() & fund['shareholders_equity'].notna() & (fund['shareholders_equity'] > 0) & fund['debt_to_equity'].isna()
fund.loc[de_mask, 'debt_to_equity'] = fund.loc[de_mask, 'total_debt'] / fund.loc[de_mask, 'shareholders_equity']

# ROE
roe_mask = brk_b_mask & fund['net_income_quarterly'].notna() & fund['shareholders_equity'].notna() & (fund['shareholders_equity'] > 0) & fund['roe'].isna()
fund.loc[roe_mask, 'roe'] = fund.loc[roe_mask, 'net_income_quarterly'] / fund.loc[roe_mask, 'shareholders_equity']

# ROIC
roic_mask = brk_b_mask & fund['ebit'].notna() & fund['total_debt'].notna() & fund['shareholders_equity'].notna() & fund['cash_and_equivalents'].notna() & fund['roic'].isna()
invested_capital = fund['total_debt'] + fund['shareholders_equity'] - fund['cash_and_equivalents']
tax_rate = np.where(
    (fund['ebit'] > 0) & (fund['net_income_quarterly'].notna()) & (fund['net_income_quarterly'] < fund['ebit']),
    1 - fund['net_income_ttm'] / fund['ebit'],   # both TTM: ebit is a TTM figure
    0.21
)
tax_rate = np.clip(tax_rate, 0, 0.5)
nopat = fund['ebit'] * (1 - tax_rate)
fund.loc[roic_mask & (invested_capital > 0), 'roic'] = nopat[roic_mask & (invested_capital > 0)] / invested_capital[roic_mask & (invested_capital > 0)]

# FCF margin
fcfm_mask = brk_b_mask & fund['free_cash_flow'].notna() & fund['revenue_quarterly'].notna() & (fund['revenue_quarterly'] > 0) & fund['fcf_margin'].isna()
fund.loc[fcfm_mask, 'fcf_margin'] = fund.loc[fcfm_mask, 'free_cash_flow'] / fund.loc[fcfm_mask, 'revenue_ttm']

# Reinvestment rate
rr_mask = brk_b_mask & fund['capital_expenditure_ttm'].notna() & fund['ebit'].notna() & (fund['ebit'] > 0) & fund['reinvestment_rate'].isna()
nopat_rr = fund['ebit'] * (1 - tax_rate)
fund.loc[rr_mask & (nopat_rr > 0), 'reinvestment_rate'] = fund.loc[rr_mask & (nopat_rr > 0), 'capital_expenditure_ttm'] / nopat_rr[rr_mask & (nopat_rr > 0)]

# EV/EBITDA
ev_mask = brk_b_mask & fund['market_cap'].notna() & fund['total_debt'].notna() & fund['cash_and_equivalents'].notna() & fund['ebit'].notna() & fund['capital_expenditure_ttm'].notna() & fund['ev_ebitda'].isna()
ev = fund['market_cap'] + fund['total_debt'] - fund['cash_and_equivalents']
ebitda_approx = fund['ebit'] + fund['capital_expenditure_ttm'].abs()
fund.loc[ev_mask & (ebitda_approx > 0), 'ev_ebitda'] = ev[ev_mask & (ebitda_approx > 0)] / ebitda_approx[ev_mask & (ebitda_approx > 0)]

# market_cap_b
fund.loc[brk_b_mask & fund['market_cap'].notna() & fund['market_cap_b'].isna(), 'market_cap_b'] = fund['market_cap'] / 1e9

# total_assets_b
fund.loc[brk_b_mask & fund['total_assets'].notna() & fund['total_assets_b'].isna(), 'total_assets_b'] = fund['total_assets'] / 1e9

# mktcap_to_assets
mta_mask = brk_b_mask & fund['market_cap'].notna() & fund['total_assets'].notna() & (fund['total_assets'] > 0) & fund['mktcap_to_assets'].isna()
fund.loc[mta_mask, 'mktcap_to_assets'] = fund.loc[mta_mask, 'market_cap'] / fund.loc[mta_mask, 'total_assets']

# Drop helper
fund = fund.drop(columns=['price_at_date'])

# Save
print("\nSaving...")
fund.to_parquet(FUND_PATH, index=False)

# Verify
brk_b_check = fund[fund['ticker'] == 'BRK-B'].sort_values('as_of_date')
print(f"BRK-B rows after fix: {len(brk_b_check)}")
print(f"Sources: {brk_b_check['source'].value_counts().to_dict()}")
latest = brk_b_check.iloc[-1]
print(f"Latest: {latest['as_of_date']}")
print(f"  Rev: {latest.get('revenue_quarterly', 'NaN')}")
print(f"  NI: {latest.get('net_income_quarterly', 'NaN')}")
print(f"  FCF: {latest.get('free_cash_flow', 'NaN')}")
print(f"  Shares: {latest.get('shares_outstanding', 'NaN')}")
print(f"  ROIC: {latest.get('roic', 'NaN')}")
print(f"  ROE: {latest.get('roe', 'NaN')}")
print(f"  D/E: {latest.get('debt_to_equity', 'NaN')}")
print(f"  FCF Margin: {latest.get('fcf_margin', 'NaN')}")
print(f"  Market Cap: {latest.get('market_cap', 'NaN')}")