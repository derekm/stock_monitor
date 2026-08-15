#!/usr/bin/env python3
"""
Damodaran Cross-Holdings Valuation - OPTIMIZED VECTORIZED VERSION

Uses actual ownership percentages (shares held / shares outstanding) from 13F-HR data.
Vectorized for speed.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

print("Loading data...")

# Load historical 13F-HR holdings (has held_shares!)
hist = pd.read_parquet('historical_13f_holdings.parquet')
print(f"Loaded {len(hist)} historical holdings")

# Load CUSIP to ticker map
with open('cusip_ticker_map.json') as f:
    cusip_to_ticker = json.load(f)

# Map CUSIP to ticker
def map_cusip_to_ticker(cusip):
    if pd.isna(cusip):
        return None
    cusip_clean = str(cusip).strip().upper()
    return cusip_to_ticker.get(cusip_clean)

hist['held_ticker'] = hist['held_cusip'].apply(map_cusip_to_ticker)
hist['as_of_date'] = pd.to_datetime(hist['as_of_date'])
hist['filing_date'] = pd.to_datetime(hist['filing_date'])

# Keep only identified tickers
hist_identified = hist.dropna(subset=['held_ticker']).copy()
print(f"Identified holdings: {len(hist_identified)} / {len(hist)} ({len(hist_identified)/len(hist)*100:.1f}%)")

# Load daily prices for market cap
prices = pd.read_parquet('daily_prices.parquet')
prices['date'] = pd.to_datetime(prices['date'])

# Load fundamentals for shares_outstanding and financials
fund = pd.read_parquet('fundamentals.parquet')
fund['as_of_date'] = pd.to_datetime(fund['as_of_date'])

# ============================================================
# VECTORIZED: BUILD SHARES OUTSTANDING TIME SERIES
# ============================================================
print("\nBuilding shares outstanding & market cap time series...")

# Get shares_outstanding from fundamentals (already has time series)
so_fund = fund[['ticker', 'as_of_date', 'shares_outstanding']].dropna()
so_fund = so_fund.sort_values(['ticker', 'as_of_date'])

# Get market_cap from fundamentals
mc_fund = fund[['ticker', 'as_of_date', 'market_cap', 'shareholders_equity']].dropna()
mc_fund = mc_fund.sort_values(['ticker', 'as_of_date'])

# ============================================================
# AGGREGATE HOLDINGS PER FILER PER QUARTER PER HELD TICKER
# ============================================================
print("\nAggregating holdings per quarter...")

agg = hist_identified.groupby(
    ['filer_ticker', 'as_of_date', 'held_ticker'],
    as_index=False
).agg({
    'held_shares': 'sum',
    'held_value_thousands': 'sum',
    'held_cusip': 'first',
    'filer_cik': 'first',
    'filing_date': 'min'
})

agg['holding_value'] = agg['held_value_thousands'] * 1000  # Convert to dollars

print(f"Aggregated to {len(agg)} unique filer-quarter-held combinations")

# ============================================================
# VECTORIZED: GET SHARES OUTSTANDING AS OF EACH DATE
# ============================================================
print("\nComputing actual ownership percentages (vectorized)...")

# For each held ticker, get the latest shares_outstanding as of each quarter
quarters = sorted(agg['as_of_date'].unique())
held_tickers = agg['held_ticker'].unique()

# Convert quarters to Timestamp for consistency
quarter_timestamps = [pd.Timestamp(q) for q in quarters]

# Build a function that merges as-of-date efficiently
def merge_asof(df_left, df_right, left_on, right_on, by=None):
    """Merge asof using pandas merge_asof"""
    left = df_left.copy()
    right = df_right.copy()
    # Ensure date columns are datetime64[ns] for compatibility
    left[left_on] = pd.to_datetime(left[left_on]).dt.floor('ns')
    right[right_on] = pd.to_datetime(right[right_on]).dt.floor('ns')
    left = left.sort_values(left_on)
    right = right.sort_values(right_on)
    if by:
        return pd.merge_asof(left, right, left_on=left_on, right_on=right_on, by=by, direction='backward')
    else:
        return pd.merge_asof(left, right, left_on=left_on, right_on=right_on, direction='backward')

# Merge shares outstanding
print("Merging shares outstanding...")
agg_so = merge_asof(
    agg[['filer_ticker', 'as_of_date', 'held_ticker', 'held_shares', 'holding_value', 'held_cusip']].copy(),
    so_fund.rename(columns={'ticker': 'held_ticker', 'shares_outstanding': 'held_shares_outstanding'}),
    left_on='as_of_date',
    right_on='as_of_date',
    by='held_ticker'
)

print(f"  After SO merge: {agg_so['held_shares_outstanding'].notna().sum()} / {len(agg_so)} have SO data")

# Merge market cap for held companies
print("Merging market cap...")
mc_fund = fund[['ticker', 'as_of_date', 'market_cap']].dropna()
mc_fund = mc_fund.rename(columns={'ticker': 'held_ticker', 'market_cap': 'held_market_cap'})
mc_fund = mc_fund.sort_values(['held_ticker', 'as_of_date'])

# Convert dates
agg_so['as_of_date'] = pd.to_datetime(agg_so['as_of_date']).astype('datetime64[ns]')
mc_fund['as_of_date'] = pd.to_datetime(mc_fund['as_of_date']).astype('datetime64[ns]')

# Merge by held_ticker and as_of_date
agg_mc = merge_asof(
    agg_so,
    mc_fund,
    left_on='as_of_date',
    right_on='as_of_date',
    by='held_ticker'
)

print(f"  After MC merge: {agg_mc['held_market_cap'].notna().sum()} / {len(agg_mc)} have MC data")

# ============================================================
# COMPUTE OWNERSHIP % AND CATEGORIES
# ============================================================
agg_mc['ownership_pct'] = agg_mc['held_shares'] / agg_mc['held_shares_outstanding']

# Cap at 100% for sanity
agg_mc['ownership_pct'] = agg_mc['ownership_pct'].clip(upper=1.0)

# Determine category
conditions = [
    agg_mc['ownership_pct'].isna(),
    agg_mc['ownership_pct'] >= 0.50,
    agg_mc['ownership_pct'] >= 0.20,
]
choices = ['UNKNOWN', 'MAJORITY_CONSOLIDATED', 'EQUITY_METHOD']
agg_mc['category'] = np.select(conditions, choices, default='MINORITY_PASSIVE')

print(f"\nClassification counts:")
print(agg_mc['category'].value_counts())

# Save actual ownership percentages
agg_mc.to_parquet('actual_ownership_percentages.parquet', index=False)
print("\nSaved actual_ownership_percentages.parquet")

# ============================================================
# DAMODARAN VALUATION PER FILER PER QUARTER (VECTORIZED)
# ============================================================
print("\n=== DAMODARAN VALUATION (Vectorized) ===")

# Load look-through fundamentals
lt_fund = pd.read_parquet('quarterly_lookthrough_fundamentals_extended.parquet')
lt_fund['as_of_date'] = pd.to_datetime(lt_fund['as_of_date'])

# Get filer fundamentals (parent company)
filers = agg_mc['filer_ticker'].unique()
filer_fund = fund[fund['ticker'].isin(filers)].copy()
filer_fund['as_of_date'] = pd.to_datetime(filer_fund['as_of_date'])

# ============================================================
# FILTER: Only operating/holding companies have meaningful cross-holdings
# Asset managers (BLK, GS, etc.) file 13F for CLIENT portfolios, not own investments
# ============================================================
asset_managers = {'BLK', 'GS', 'MS', 'JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC', 'PRU', 
                  'MET', 'COF', 'AXP', 'ALL', 'TRV', 'AIG', 'CB', 'CINF', 'AFG', 'WRB', 'FAF'}
operating_filers = [f for f in filers if f not in asset_managers]

print(f"\nTotal filers: {len(filers)}")
print(f"Asset managers (excluded): {len(asset_managers & set(filers))}")
print(f"Operating companies (analyzed): {len(operating_filers)}")
print(f"Operating: {operating_filers}")

valuation_results = []

for q, q_ts in tqdm(zip(quarters, quarter_timestamps), total=len(quarters), desc="Valuing filers"):
    q_own = agg_mc[agg_mc['as_of_date'] == q_ts]
    q_lt = lt_fund[lt_fund['as_of_date'] == q_ts]
    q_filer_fund = filer_fund[filer_fund['as_of_date'] <= q_ts].drop_duplicates('ticker', keep='last')
    q_filer_fund = q_filer_fund.set_index('ticker')
    
    # Get parent market caps for operating filers only
    parent_mcs = {}
    for filer in operating_filers:
        if filer in q_filer_fund.index:
            pf = q_filer_fund.loc[filer]
            mc = pf.get('market_cap', np.nan)
            if pd.isna(mc) or mc == 0:
                shares = pf.get('shares_outstanding', np.nan)
                f_prices = prices[prices['date'] <= q_ts]
                f_prices = f_prices[f_prices['ticker'] == filer]
                if len(f_prices) > 0:
                    latest_price = f_prices.iloc[-1]['adj_close']
                    if pd.notna(shares) and shares > 0 and pd.notna(latest_price):
                        mc = shares * latest_price
            if pd.notna(mc) and mc > 0:
                parent_mcs[filer] = mc
    
    if not parent_mcs:
        continue
    
    # Process each filer
    for filer, parent_market_cap in parent_mcs.items():
        f_own = q_own[q_own['filer_ticker'] == filer]
        if len(f_own) == 0:
            continue
        
        # Separate holdings by category
        majority = f_own[f_own['category'] == 'MAJORITY_CONSOLIDATED']
        equity_method = f_own[f_own['category'] == 'EQUITY_METHOD']
        minority = f_own[f_own['category'] == 'MINORITY_PASSIVE']
        unknown = f_own[f_own['category'] == 'UNKNOWN']
        
        # ============================================================
        # DAMODARAN VALUATION LOGIC
        # ============================================================
        
        # 1. MAJORITY (CONSOLIDATED) HOLDINGS
        # Parent market cap already includes 100% of these subs (consolidated)
        # We subtract the Minority Interest (NCI) = portion we DON'T own
        minority_interest_value = 0
        for _, h in majority.iterrows():
            sub = h['held_ticker']
            pct = h['ownership_pct']
            
            sub_mc = h['held_market_cap']
            if pd.isna(sub_mc) or sub_mc == 0:
                continue
            
            # Use held_equity directly if available, otherwise estimate from mc
            sub_equity = h.get('held_equity', 0)
            if pd.isna(sub_equity) or sub_equity == 0:
                sub_equity = sub_mc * 0.5  # rough estimate: equity ~ 50% of market cap
            
            # NCI = (1 - pct) * sub_equity
            minority_interest_value += (1 - pct) * sub_equity
        
        # 2. EQUITY METHOD (20-50%) HOLDINGS
        # Not consolidated - add our % of sub equity
        equity_method_value = 0
        for _, h in equity_method.iterrows():
            sub = h['held_ticker']
            pct = h['ownership_pct']
            
            sub_mc = h['held_market_cap']
            if pd.isna(sub_mc) or sub_mc == 0:
                continue
            
            sub_equity = h.get('held_equity', 0)
            if pd.isna(sub_equity) or sub_equity == 0:
                sub_equity = sub_mc * 0.5
            
            equity_method_value += pct * sub_equity
        
        # 3. MINORITY PASSIVE (<20%) HOLDINGS
        # 13F holding_value is already the market value of OUR stake (our % * sub_mkt_cap)
        # Use it directly as our cross-holding value
        minority_holding_value = minority['holding_value'].sum()
        unknown_value = unknown['holding_value'].sum()
        
        # Cross holdings value = equity method + minority (NOT majority, already in parent mkt cap)
        cross_holdings_value = equity_method_value + minority_holding_value + unknown_value
        
        # Damodaran: Value of Equity = Parent Market Cap - Minority Interest + Cross Holdings
        damodaran_equity = parent_market_cap - minority_interest_value + cross_holdings_value
        lookthrough_equity = parent_market_cap + cross_holdings_value  # naive: add everything
        
        valuation_results.append({
            'as_of_date': q,
            'filer_ticker': filer,
            'parent_market_cap': parent_market_cap,
            'n_majority': len(majority),
            'n_equity_method': len(equity_method),
            'n_minority': len(minority),
            'n_unknown': len(unknown),
            'equity_method_value': equity_method_value,
            'minority_value': minority_holding_value,
            'unknown_value': unknown_value,
            'cross_holdings_total': cross_holdings_value,
            'minority_interest_subtracted': minority_interest_value,
            'damodaran_equity_value': damodaran_equity,
            'lookthrough_equity_value': lookthrough_equity,
            'naive_equity_value': parent_market_cap,
            'cross_holding_impact': damodaran_equity - parent_market_cap,
            'nci_estimate': minority_interest_value,
        })

val_df = pd.DataFrame(valuation_results)
val_df.to_parquet('damodaran_crossholdings_valuation.parquet', index=False)
print(f"\nSaved {len(val_df)} valuation rows to damodaran_crossholdings_valuation.parquet")

# ============================================================
# SUMMARY STATS
# ============================================================
print("\n=== VALUATION SUMMARY (Latest Quarter) ===")
if len(val_df) > 0:
    latest_q = val_df['as_of_date'].max()
    latest = val_df[val_df['as_of_date'] == latest_q]

    for _, row in latest.sort_values('damodaran_equity_value', ascending=False).head(20).iterrows():
        print(f"{row['filer_ticker']}: "
              f"Parent=${row['parent_market_cap']/1e9:.1f}B, "
              f"Cross=${row['cross_holdings_total']/1e9:.1f}B, "
              f"MinorityInt=${row['minority_interest_subtracted']/1e9:.1f}B, "
              f"Damodaran=${row['damodaran_equity_value']/1e9:.1f}B, "
              f"Maj={row['n_majority']}, EqM={row['n_equity_method']}, Min={row['n_minority']}")

    # Show impact
    print("\n=== CROSS-HOLDING IMPACT ===")
    impact = latest[['filer_ticker', 'parent_market_cap', 'damodaran_equity_value', 'cross_holding_impact']].copy()
    impact['pct_change'] = (impact['damodaran_equity_value'] - impact['parent_market_cap']) / impact['parent_market_cap'] * 100
    print(impact.sort_values('pct_change', ascending=False).to_string(index=False))

    # Category breakdown over time
    print("\n=== CATEGORY TRENDS ===")
    cat_trend = agg_mc.groupby(['as_of_date', 'category']).size().unstack(fill_value=0)
    print(cat_trend.tail(10).to_string())
else:
    print("No valuation results generated!")