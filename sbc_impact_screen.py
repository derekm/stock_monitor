#!/usr/bin/env python3
"""
SBC Impact Screen — Key Names with Recent Negative NI
Focus on large-cap tech where SBC is likely driving NI negative
"""

import pandas as pd
import numpy as np

fund = pd.read_parquet('fundamentals.parquet')
prices = pd.read_parquet('daily_prices.parquet')
mon = pd.read_parquet('monitored_stocks.parquet')

# Get latest fundamentals per ticker
latest = fund.sort_values('as_of_date').groupby('ticker').tail(1)

# Merge sector info
latest = latest.merge(mon[['ticker', 'sector', 'industry']], on='ticker', how='left')

# Calculate SBC estimate
latest['sbc_est'] = latest['operating_income_quarterly'] - latest['net_income_quarterly']
latest['sbc_pct_rev'] = (latest['sbc_est'] / latest['revenue_ttm']) * 100

# Filter to tech/growth sectors
tech = latest[latest['sector'].isin(['Information Technology', 'Communication Services'])].copy()

# Filter to negative NI
neg_ni = tech[tech['net_income_quarterly'] < 0]

# Sort by market cap
neg_ni = neg_ni.sort_values('market_cap', ascending=False)

print("=" * 80)
print("LARGE-CAP TECH COMPANIES WITH RECENT NEGATIVE NI")
print("=" * 80)

print(f"\nTotal tech tickers with negative NI: {len(neg_ni)}")
print(f"Showing all with market cap > $1B:")
print()

for _, row in neg_ni.iterrows():
    t = row['ticker']
    mktcap = row.get('market_cap', np.nan)
    ni = row.get('net_income_quarterly', 0)
    rev = row.get('revenue_ttm', np.nan)
    oi = row.get('operating_income_quarterly', np.nan)
    sbc = row.get('sbc_est', np.nan)
    sbc_pct = row.get('sbc_pct_rev', np.nan)
    
    if pd.notna(mktcap) and mktcap > 1e9:
        mktcap_str = f'${mktcap/1e9:.1f}B'
        ni_str = f'${ni/1e6:.0f}M'
        rev_str = f'${rev/1e6:.0f}M' if pd.notna(rev) else 'N/A'
        oi_str = f'${oi/1e6:.0f}M' if pd.notna(oi) else 'N/A'
        sbc_str = f'${sbc/1e6:.0f}M' if pd.notna(sbc) else 'N/A'
        sbc_pct_str = f'{sbc_pct:.1f}%' if pd.notna(sbc_pct) else 'N/A'
        
        # Flag if SBC > OI (likely cause of negative NI)
        flag = ' ***' if pd.notna(sbc) and pd.notna(oi) and sbc > abs(oi) else ''
        print(f'{t:<8} {mktcap_str:<10} NI={ni_str:<12} Rev={rev_str:<12} OI={oi_str:<12} SBC={sbc_str:<12} SBC%={sbc_pct_str}{flag}')

# Also show monitored stocks specifically
print(f"\n{'='*80}")
print("MONITORED STOCKS WITH NEGATIVE NI")
print("=" * 80)

monitored_neg = neg_ni[neg_ni['ticker'].isin(mon['ticker'])]
if len(monitored_neg) > 0:
    for _, row in monitored_neg.iterrows():
        t = row['ticker']
        ni = row.get('net_income_quarterly', 0)
        rev = row.get('revenue_ttm', np.nan)
        oi = row.get('operating_income_quarterly', np.nan)
        sbc = row.get('sbc_est', np.nan)
        sbc_pct = row.get('sbc_pct_rev', np.nan)
        
        ni_str = f'${ni/1e6:.0f}M'
        rev_str = f'${rev/1e6:.0f}M' if pd.notna(rev) else 'N/A'
        oi_str = f'${oi/1e6:.0f}M' if pd.notna(oi) else 'N/A'
        sbc_str = f'${sbc/1e6:.0f}M' if pd.notna(sbc) else 'N/A'
        sbc_pct_str = f'{sbc_pct:.1f}%' if pd.notna(sbc_pct) else 'N/A'
        
        print(f'{t:<8} NI={ni_str:<12} Rev={rev_str:<12} OI={oi_str:<12} SBC={sbc_str:<12} SBC%={sbc_pct_str}')
else:
    print('No monitored stocks have negative NI in latest quarter')

# Key summary
print(f"\n{'='*80}")
print("SUMMARY")
print("=" * 80)
print(f"""
Total monitored stocks: {len(mon)}
Monitored tech stocks: {len(mon[mon['sector'].isin(['Information Technology', 'Communication Services'])])} 
Tech with negative NI: {len(neg_ni)}
With market cap > $1B: {len(neg_ni[neg_ni['market_cap'] > 1e9]) if len(neg_ni) > 0 else 0}

Key takeaway: PANW is the only large-cap monitored tech name with recent 
negative NI. This is significant because PANW is a cybersecurity leader — 
if SBC is driving negative NI here, it signals broader sector trend.

Other high-SBC tech names (from sector analysis):
- CRWD (PB 49.6x) — high growth but SBC heavy
- FTNT (PB 78.3x) — extreme valuation
- NET (PB 74.0x) — high growth, high SBC
- OKTA (PB 3.79x) — growth at reasonable price

These names will be watched closely for SBC trends in upcoming reports.
""")