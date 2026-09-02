#!/usr/bin/env python3
"""
Compute missing derived metrics: earnings_stability, interest_coverage
and fill any remaining gaps in fundamentals.parquet
"""

import pandas as pd
from analytics_common import atomic_write_parquet
import numpy as np
from pathlib import Path

FUND_PATH = Path('fundamentals.parquet')

print("Loading fundamentals...")
fund = pd.read_parquet(FUND_PATH)
from datetime import date as _date, datetime as _dt

def _as_date(x):
    if isinstance(x, _date) and not isinstance(x, _dt):
        return x
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x.date()
    if hasattr(x, "date"):
        return x.date()
    return pd.Timestamp(x).date()

if "as_of_date" in fund.columns:
    fund["as_of_date"] = fund["as_of_date"].map(_as_date)
print(f"Loaded: {len(fund)} rows, {fund['ticker'].nunique()} tickers")

# ============================================================
# 0. REVENUE_TTM from quarterly (PIT) then ffill
# ============================================================
print("\n=== Filling revenue_ttm ===")
fund = fund.sort_values(["ticker", "as_of_date"]).reset_index(drop=True)
if "revenue_quarterly" in fund.columns:
    q = pd.to_numeric(fund["revenue_quarterly"], errors="coerce")
    sub = fund.loc[q.notna(), ["ticker", "as_of_date"]].copy()
    sub["qrev"] = q[q.notna()].to_numpy()
    sub["_ts"] = pd.to_datetime(sub["as_of_date"])
    g = sub.groupby("ticker", sort=False)
    sub["ttm4"] = g["qrev"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    sub["d0"] = g["_ts"].shift(3)
    span = (sub["_ts"] - sub["d0"]).dt.days
    ok = sub["ttm4"].notna() & (span >= 240) & (span <= 400)
    ttm_map = sub.loc[ok, ["ticker", "as_of_date", "ttm4"]]
    fund = fund.merge(ttm_map, on=["ticker", "as_of_date"], how="left")
    need = fund["revenue_ttm"].isna() & fund["ttm4"].notna()
    print(f"Filled revenue_ttm from 4q rolling: {int(need.sum())}")
    fund.loc[need, "revenue_ttm"] = fund.loc[need, "ttm4"]
    fund.drop(columns=["ttm4"], inplace=True)
fund["revenue_ttm"] = fund.groupby("ticker")["revenue_ttm"].ffill()
if "free_cash_flow" in fund.columns:
    fund["free_cash_flow"] = fund.groupby("ticker")["free_cash_flow"].ffill()
if "revenue_quarterly" in fund.columns:
    fund["revenue_quarterly"] = fund.groupby("ticker")["revenue_quarterly"].ffill()
print(f"revenue_ttm coverage now {fund['revenue_ttm'].notna().mean():.1%} rows")

# ============================================================
# 1. COMPUTE EARNINGS STABILITY
# ============================================================
print("\n=== Computing Earnings Stability ===")

# Earnings stability = R-squared of quarterly net income trend over rolling window
# Or: correlation of actual vs trend earnings
# Standard Damodaran: earnings_stability = 1 - (std of earnings changes / mean earnings)

def compute_earnings_stability(group):
    """Compute earnings stability for a ticker's quarterly data"""
    group = group.sort_values('as_of_date')
    ni = group['net_income_quarterly'].dropna()
    if len(ni) < 8:  # Need minimum quarters
        return pd.Series(index=group.index, dtype=float)
    
    # Compute rolling stability over trailing 12 quarters
    stability = pd.Series(index=group.index, dtype=float)
    
    for i in range(len(group)):
        row = group.iloc[i]
        date = row['as_of_date']
        # Get NI up to this date
        hist_ni = group[group['as_of_date'] <= date]['net_income_quarterly'].dropna()
        if len(hist_ni) >= 8:
            # Coefficient of variation inverse (higher = more stable)
            # Or R-squared of linear trend
            x = np.arange(len(hist_ni))
            y = hist_ni.values
            if np.std(y) > 0:
                # R-squared of linear trend
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                if ss_tot > 0:
                    r2 = 1 - ss_res / ss_tot
                    stability.iloc[i] = max(0, min(1, r2))  # Clamp 0-1
    
    return stability

# Apply by ticker
print("Computing earnings stability by ticker...")
stability_results = fund.groupby('ticker', group_keys=False).apply(compute_earnings_stability)
fund['earnings_stability_computed'] = stability_results
print(f"Computed earnings_stability for {fund['earnings_stability_computed'].notna().sum()} rows")

# Fill existing earnings_stability where missing
mask = fund['earnings_stability'].isna() & fund['earnings_stability_computed'].notna()
fund.loc[mask, 'earnings_stability'] = fund.loc[mask, 'earnings_stability_computed']
print(f"Filled {mask.sum()} earnings_stability values")

# ============================================================
# 2. COMPUTE INTEREST COVERAGE
# ============================================================
print("\n=== Computing Interest Coverage ===")

# Interest coverage = EBIT / Interest Expense
# Need interest expense from EDGAR
# Check if we have interest_expense column or can derive

if 'interest_expense' in fund.columns:
    print("interest_expense column exists")
else:
    print("No interest_expense column - need to derive from EDGAR raw data")
    # For now, estimate from debt * 5% for companies with debt
    # Better: fetch from EDGAR companyfacts for key tickers

# Estimate interest coverage where we have EBIT and debt
# interest_coverage = EBIT / (Debt * estimated_rate)
estimated_rate = 0.05  # 5% average cost of debt

mask_ic = (
    fund['ebit'].notna() & 
    fund['total_debt'].notna() & 
    (fund['total_debt'] > 0) & 
    fund['interest_coverage'].isna()
)
fund.loc[mask_ic, 'interest_coverage_est'] = fund.loc[mask_ic, 'ebit'] / (fund.loc[mask_ic, 'total_debt'] * estimated_rate)
print(f"Estimated interest_coverage for {mask_ic.sum()} rows")

# Fill existing
mask = fund['interest_coverage'].isna() & fund['interest_coverage_est'].notna()
fund.loc[mask, 'interest_coverage'] = fund.loc[mask, 'interest_coverage_est']
print(f"Filled {mask.sum()} interest_coverage values")

# ============================================================
# 3. RE-COMPUTE ALL DERIVED METRICS WITH FULL DATA
# ============================================================
print("\n=== Re-computing all derived metrics ===")

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

# ROE
roe_mask = fund['net_income_quarterly'].notna() & fund['shareholders_equity'].notna() & (fund['shareholders_equity'] > 0) & fund['roe'].isna()
fund.loc[roe_mask, 'roe'] = fund.loc[roe_mask, 'net_income_quarterly'] / fund.loc[roe_mask, 'shareholders_equity']
print(f"ROE: {roe_mask.sum()}")

# D/E
de_mask = fund['total_debt'].notna() & fund['shareholders_equity'].notna() & (fund['shareholders_equity'] > 0) & fund['debt_to_equity'].isna()
fund.loc[de_mask, 'debt_to_equity'] = fund.loc[de_mask, 'total_debt'] / fund.loc[de_mask, 'shareholders_equity']
print(f"D/E: {de_mask.sum()}")

# FCF Margin — TTM FCF / TTM revenue (never quarterly)
fcfm_mask = fund['free_cash_flow'].notna() & fund['revenue_ttm'].notna() & (fund['revenue_ttm'] > 0) & fund['fcf_margin'].isna()
fund.loc[fcfm_mask, 'fcf_margin'] = fund.loc[fcfm_mask, 'free_cash_flow'] / fund.loc[fcfm_mask, 'revenue_ttm']
print(f"FCF Margin: {fcfm_mask.sum()}")

# Reinvestment Rate
rr_mask = fund['capital_expenditure_ttm'].notna() & fund['ebit'].notna() & (fund['ebit'] > 0) & fund['reinvestment_rate'].isna()
nopat_rr = fund['ebit'] * (1 - tax_rate)
fund.loc[rr_mask & (nopat_rr > 0), 'reinvestment_rate'] = fund.loc[rr_mask & (nopat_rr > 0), 'capital_expenditure_ttm'] / nopat_rr[rr_mask & (nopat_rr > 0)]
print(f"Reinvestment Rate: {(rr_mask & (nopat_rr > 0)).sum()}")

# EV/EBITDA
ev_mask = (
    fund['market_cap'].notna() & 
    fund['total_debt'].notna() & 
    fund['cash_and_equivalents'].notna() & 
    fund['ebitda'].notna() &   # real EBITDA column (220k rows), not ebit+capex proxy
    fund['ev_ebitda'].isna()
)
ev = fund['market_cap'] + fund['total_debt'] - fund['cash_and_equivalents']
fund.loc[ev_mask & (fund['ebitda'] > 0), 'ev_ebitda'] = ev[ev_mask & (fund['ebitda'] > 0)] / fund['ebitda'][ev_mask & (fund['ebitda'] > 0)]
print(f"EV/EBITDA (real ebitda): {(ev_mask & (fund['ebitda'] > 0)).sum()}")

# PB Ratio
pb_mask = fund['market_cap'].notna() & fund['shareholders_equity'].notna() & (fund['shareholders_equity'] > 0) & fund['pb_ratio'].isna()
fund.loc[pb_mask, 'pb_ratio'] = fund.loc[pb_mask, 'market_cap'] / fund.loc[pb_mask, 'shareholders_equity']
print(f"PB Ratio: {pb_mask.sum()}")

# Market Cap to Assets
mta_mask = fund['market_cap'].notna() & fund['total_assets'].notna() & (fund['total_assets'] > 0) & fund['mktcap_to_assets'].isna()
fund.loc[mta_mask, 'mktcap_to_assets'] = fund.loc[mta_mask, 'market_cap'] / fund.loc[mta_mask, 'total_assets']
print(f"Mktcap/Assets: {mta_mask.sum()}")

# Billions columns
fund.loc[fund['market_cap'].notna() & fund['market_cap_b'].isna(), 'market_cap_b'] = fund['market_cap'] / 1e9
fund.loc[fund['total_assets'].notna() & fund['total_assets_b'].isna(), 'total_assets_b'] = fund['total_assets'] / 1e9

# ============================================================
# 4. CLEAN UP AND SAVE
# ============================================================
print("\n=== Cleaning up ===")
# Drop helper columns
for col in ['earnings_stability_computed', 'interest_coverage_est']:
    if col in fund.columns:
        fund = fund.drop(columns=[col])

# Sort and deduplicate
fund = fund.sort_values(['ticker', 'as_of_date']).drop_duplicates(
    subset=['ticker', 'as_of_date'], keep='first'
)

# Save
print("Saving...")
atomic_write_parquet(fund, FUND_PATH)
print(f"Saved: {len(fund)} rows, {fund['ticker'].nunique()} tickers")

# ============================================================
# VERIFICATION
# ============================================================
latest = fund.sort_values('as_of_date').groupby('ticker').tail(1)
print("\n=== FINAL COVERAGE (Latest Quarter) ===")
for col in ['roic', 'roe', 'debt_to_equity', 'fcf_margin', 'reinvestment_rate', 
            'ev_ebitda', 'pb_ratio', 'market_cap', 'shares_outstanding',
            'market_cap_b', 'total_assets_b', 'mktcap_to_assets', 'revenue_quarterly',
            'net_income_quarterly', 'free_cash_flow', 'ebit', 'total_debt', 'cash_and_equivalents',
            'interest_coverage', 'earnings_stability']:
    if col in latest.columns:
        cnt = latest[col].notna().sum()
        print(f"  {col}: {cnt}/{len(latest)} ({cnt/len(latest)*100:.1f}%)")

print("\n=== BY SOURCE ===")
for src in ['edgar', 'yfinance', 'yfinance_history']:
    src_latest = latest[latest['source'] == src]
    if len(src_latest) > 0:
        print(f"  {src}: {len(src_latest)} tickers")
        for col in ['roic', 'roe', 'fcf_margin', 'market_cap', 'shares_outstanding', 
                    'interest_coverage', 'earnings_stability', 'reinvestment_rate']:
            if col in src_latest.columns:
                cnt = src_latest[col].notna().sum()
                print(f"    {col}: {cnt}/{len(src_latest)} ({cnt/len(src_latest)*100:.1f}%)")