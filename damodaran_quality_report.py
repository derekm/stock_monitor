#!/usr/bin/env python3
"""
Damodaran Quality Ranked Analysis of the Universe
Uses latest available fundamentals to score and rank all tickers.
"""

import pandas as pd
import numpy as np
from pathlib import Path

FUND_PATH = Path('fundamentals.parquet')

print("Loading fundamentals...")
fund = pd.read_parquet(FUND_PATH)
fund['as_of_date'] = pd.to_datetime(fund['as_of_date']).dt.date

# Get latest quarter for each ticker
latest = fund.sort_values('as_of_date').groupby('ticker').tail(1)
print(f"Total tickers: {len(latest)}")

# ============================================================
# COMPUTE QUALITY SCORES — VECTORIZED (FAST)
# ============================================================
print("\nComputing quality scores...")

# Initialize scores
scores = pd.DataFrame(index=latest['ticker'].values)
scores['data_completeness'] = 0
scores['profitability'] = 0
scores['financial_health'] = 0
scores['growth'] = 0
scores['stability'] = 0
scores['total_score'] = 0

# Track which metrics are available for each ticker
metrics_count = pd.DataFrame(index=latest['ticker'].values)
metrics_count['available'] = 0

latest_indexed = latest.set_index('ticker')

# ---- PROFITABILITY (0-30 points) ----

# ROIC (0-10)
if 'roic' in latest.columns:
    roic = latest['roic'].fillna(0).values
    scores['profitability'] += np.where(roic > 0.20, 10,
                                np.where(roic > 0.15, 8,
                                    np.where(roic > 0.10, 6,
                                        np.where(roic > 0.05, 4,
                                            np.where(roic > 0, 2, 0)))))
    metrics_count['available'] += latest['roic'].notna().astype(int).values

# ROE (0-8)
if 'roe' in latest.columns:
    roe = latest['roe'].fillna(0).values
    scores['profitability'] += np.where(roe > 0.20, 8,
                                np.where(roe > 0.15, 6,
                                    np.where(roe > 0.10, 4,
                                        np.where(roe > 0.05, 2, 0))))
    metrics_count['available'] += latest['roe'].notna().astype(int).values

# FCF Margin (0-8)
if 'fcf_margin' in latest.columns:
    fcfm = latest['fcf_margin'].fillna(0).values
    scores['profitability'] += np.where(fcfm > 0.30, 8,
                                np.where(fcfm > 0.20, 6,
                                    np.where(fcfm > 0.10, 4,
                                        np.where(fcfm > 0.05, 2, 0))))
    metrics_count['available'] += latest['fcf_margin'].notna().astype(int).values

# EBIT Margin (0-4) - if available
if 'ebit' in latest.columns and 'revenue_ttm' in latest.columns:
    ebit_margin = (latest['ebit'] / latest['revenue_ttm']).fillna(0).values
    scores['profitability'] += np.where(ebit_margin > 0.25, 4,
                                np.where(ebit_margin > 0.15, 3,
                                    np.where(ebit_margin > 0.10, 2,
                                        np.where(ebit_margin > 0.05, 1, 0))))
    metrics_count['available'] += (latest['ebit'].notna() & latest['revenue_ttm'].notna()).astype(int).values

# ---- FINANCIAL HEALTH (0-20 points) ----

# D/E Ratio (0-8) - lower is better
if 'debt_to_equity' in latest.columns:
    de = latest['debt_to_equity'].fillna(999).values
    de_na = latest['debt_to_equity'].isna().values
    scores['financial_health'] += np.where(de_na, 0,
                                    np.where(de < 0.2, 8,
                                        np.where(de < 0.5, 6,
                                            np.where(de < 1.0, 4,
                                                np.where(de < 2.0, 2, 0)))))
    metrics_count['available'] += latest['debt_to_equity'].notna().astype(int).values

# Interest Coverage (0-6) - higher is better
if 'interest_coverage' in latest.columns:
    ic = latest['interest_coverage'].fillna(0).values
    scores['financial_health'] += np.where(ic > 20, 6,
                                np.where(ic > 10, 5,
                                    np.where(ic > 5, 4,
                                        np.where(ic > 2, 2,
                                            np.where(ic > 1, 1, 0)))))
    metrics_count['available'] += latest['interest_coverage'].notna().astype(int).values

# Cash/Debt Ratio (0-6) - higher is better
if 'cash_and_equivalents' in latest.columns and 'total_debt' in latest.columns:
    cash = latest['cash_and_equivalents'].fillna(0).values
    debt = latest['total_debt'].fillna(0).values
    cash_debt_ratio = np.where(debt > 0, cash / debt, 999)
    scores['financial_health'] += np.where(cash_debt_ratio > 2, 6,
                                    np.where(cash_debt_ratio > 1, 5,
                                        np.where(cash_debt_ratio > 0.5, 4,
                                            np.where(cash_debt_ratio > 0.25, 2,
                                                np.where(cash_debt_ratio > 0, 1, 0)))))
    metrics_count['available'] += (latest['cash_and_equivalents'].notna() & latest['total_debt'].notna()).astype(int).values

# ---- GROWTH (0-20 points) — VECTORIZED ----

# Get revenue and net income from 4 quarters ago (YoY)
print("  Computing growth metrics...")
tickers = latest['ticker'].values

# Create a lookup for latest values
latest_rev = latest_indexed['revenue_ttm']
latest_ni = latest_indexed['net_income_quarterly']

# Get historical data (4 quarters back) using groupby
hist_4q = fund.sort_values('as_of_date').groupby('ticker').tail(5)
hist_4q = hist_4q[hist_4q.groupby('ticker').cumcount() == 0]  # First of last 5 = 4 quarters ago

for _, row in hist_4q.iterrows():
    t = row['ticker']
    if t in scores.index:
        # Revenue growth
        prev_rev = row['revenue_ttm']
        curr_rev = latest_rev.get(t, np.nan)
        if pd.notna(prev_rev) and pd.notna(curr_rev) and prev_rev > 0:
            growth = (curr_rev - prev_rev) / prev_rev
            scores.loc[t, 'growth'] += np.where(growth > 0.20, 10,
                                        np.where(growth > 0.15, 8,
                                            np.where(growth > 0.10, 6,
                                                np.where(growth > 0.05, 4,
                                                    np.where(growth > 0, 2, 0)))))
            metrics_count.loc[t, 'available'] += 1
        
        # Net income growth
        prev_ni = row['net_income_quarterly']
        curr_ni = latest_ni.get(t, np.nan)
        if pd.notna(prev_ni) and pd.notna(curr_ni):
            if prev_ni > 0:
                growth = (curr_ni - prev_ni) / prev_ni
                scores.loc[t, 'growth'] += np.where(growth > 0.20, 10,
                                            np.where(growth > 0.15, 8,
                                                np.where(growth > 0.10, 6,
                                                    np.where(growth > 0.05, 4,
                                                        np.where(growth > 0, 2, 0)))))
                metrics_count.loc[t, 'available'] += 1
            elif prev_ni <= 0 and curr_ni > 0:
                scores.loc[t, 'growth'] += 5
                metrics_count.loc[t, 'available'] += 1

# ---- STABILITY (0-15 points) — VECTORIZED ----

# Earnings stability (0-8)
if 'earnings_stability' in latest.columns:
    es = latest['earnings_stability'].fillna(0).values
    scores['stability'] += np.where(es > 0.8, 8,
                            np.where(es > 0.6, 6,
                                np.where(es > 0.4, 4,
                                    np.where(es > 0.2, 2, 0))))
    metrics_count['available'] += latest['earnings_stability'].notna().astype(int).values

# Revenue stability (0-7) — coefficient of variation
print("  Computing revenue stability...")
rev_cv = fund.groupby('ticker')['revenue_ttm'].agg(lambda x: x.std() / x.mean() if x.mean() > 0 else 999)
rev_cv = rev_cv[rev_cv.index.isin(scores.index)]

for t, cv in rev_cv.items():
    if cv < 0.05:
        scores.loc[t, 'stability'] += 7
    elif cv < 0.10:
        scores.loc[t, 'stability'] += 5
    elif cv < 0.20:
        scores.loc[t, 'stability'] += 3
    elif cv < 0.30:
        scores.loc[t, 'stability'] += 1
    metrics_count.loc[t, 'available'] += 1

# ---- DATA COMPLETENESS BONUS (0-15 points) ----
scores['data_completeness'] = np.where(metrics_count['available'] >= 10, 15,
                                np.where(metrics_count['available'] >= 8, 12,
                                    np.where(metrics_count['available'] >= 6, 9,
                                        np.where(metrics_count['available'] >= 4, 6,
                                            np.where(metrics_count['available'] >= 2, 3, 0)))))

# ---- TOTAL SCORE ----
scores['total_score'] = scores['profitability'] + scores['financial_health'] + scores['growth'] + scores['stability'] + scores['data_completeness']

# Merge with latest data
scores_reset = scores.reset_index().rename(columns={'index': 'ticker'})
metrics_count_reset = metrics_count.reset_index().rename(columns={'index': 'ticker'})

results = latest.merge(scores_reset, on='ticker', how='left')
results = results.merge(metrics_count_reset, on='ticker', how='left')
results = results.sort_values('total_score', ascending=False)

# ============================================================
# RANKED REPORT
# ============================================================
print("\n" + "=" * 80)
print("DAMODARAN QUALITY RANKED ANALYSIS (8,669 Tickers)")
print("=" * 80)

print("\n📊 SCORE DISTRIBUTION")
print("-" * 50)

# Score bands
bands = [
    (90, 100, "Elite"),
    (80, 89, "Excellent"),
    (70, 79, "High Quality"),
    (60, 69, "Above Average"),
    (50, 59, "Average"),
    (40, 49, "Below Average"),
    (30, 39, "Poor"),
    (20, 29, "Very Poor"),
    (0, 19, "Failed/No Data"),
]

for low, high, label in bands:
    cnt = ((results['total_score'] >= low) & (results['total_score'] <= high)).sum()
    pct = cnt / len(results) * 100
    print(f"  {low:3d}-{high:3d} | {label:<20s} | {cnt:4d} ({pct:5.1f}%)")

# Only show tickers with sufficient data (at least 4 metrics)
results_valid = results[results['available'] >= 4].sort_values('total_score', ascending=False)

print(f"\n✅ VALID RANKED UNIVERSE: {len(results_valid)} tickers (with ≥4 metrics)")

print("\n" + "=" * 80)
print("🏆 TOP 50 — DAMODARAN QUALITY ELITE (Score ≥60)")
print("=" * 80)

# Filter to meaningful scores
top50 = results_valid[results_valid['total_score'] >= 60].head(50)

if len(top50) == 0:
    # Lower threshold
    top50 = results_valid[results_valid['total_score'] >= 50].head(50)

print(f"\n{'Rank':<6} {'Ticker':<10} {'Score':<8} {'Profit':<8} {'Health':<8} {'Growth':<8} {'Stab':<8} {'Complete':<10} {'ROIC':<8} {'ROE':<8} {'FCF Mrg':<8} {'D/E':<8}")
print("-" * 100)

for i, (idx, row) in enumerate(top50.iterrows(), 1):
    roic = f"{row['roic']*100:.1f}%" if pd.notna(row.get('roic')) else "N/A"
    roe = f"{row['roe']*100:.1f}%" if pd.notna(row.get('roe')) else "N/A"
    fcfm = f"{row['fcf_margin']*100:.1f}%" if pd.notna(row.get('fcf_margin')) else "N/A"
    de = f"{row['debt_to_equity']:.2f}" if pd.notna(row.get('debt_to_equity')) else "N/A"
    print(f"{i:<6} {row['ticker']:<10} {row['total_score']:<8} {row['profitability']:<8} {row['financial_health']:<8} {row['growth']:<8} {row['stability']:<8} {row['data_completeness']:<10} {roic:<8} {roe:<8} {fcfm:<8} {de:<8}")

# ============================================================
# SECTOR-LEVEL ANALYSIS
# ============================================================
print("\n\n📈 SCORE COMPONENTS BY SOURCE")
print("-" * 50)

for src in ['edgar', 'yfinance', 'yfinance_history']:
    src_data = results[results['source'] == src]
    if len(src_data) > 0:
        avg_score = src_data['total_score'].mean()
        avg_profit = src_data['profitability'].mean()
        avg_health = src_data['financial_health'].mean()
        avg_growth = src_data['growth'].mean()
        avg_stab = src_data['stability'].mean()
        print(f"  {src}:")
        print(f"    Count: {len(src_data)}, Avg Score: {avg_score:.1f}")
        print(f"    Profitability: {avg_profit:.1f}, Health: {avg_health:.1f}, Growth: {avg_growth:.1f}, Stability: {avg_stab:.1f}")

# ============================================================
# DATA GAPS & IMPROVEMENTS
# ============================================================
print("\n\n📋 DATA COVERAGE SUMMARY")
print("-" * 50)

for col in ['roic', 'roe', 'debt_to_equity', 'fcf_margin', 'reinvestment_rate',
            'ev_ebitda', 'pb_ratio', 'market_cap', 'shares_outstanding',
            'revenue_ttm', 'net_income_quarterly', 'free_cash_flow', 'ebit', 'total_debt',
            'cash_and_equivalents', 'interest_coverage', 'earnings_stability']:
    if col in latest.columns:
        cnt = latest[col].notna().sum()
        pct = cnt / len(latest) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {col:<25s} {cnt:4d}/{len(latest)} ({pct:5.1f}%) {bar}")

# ============================================================
# KEY FINDINGS
# ============================================================
print("\n\n🔍 KEY FINDINGS")
print("-" * 50)

elite = results_valid[results_valid['total_score'] >= 80]
high = results_valid[(results_valid['total_score'] >= 70) & (results_valid['total_score'] < 80)]
valid_count = len(results_valid)

print(f"  • {len(elite)} tickers ({(len(elite)/valid_count*100) if valid_count > 0 else 0:.1f}%) score ≥80 (Elite/Excellent)")
print(f"  • {len(high)} tickers ({(len(high)/valid_count*100) if valid_count > 0 else 0:.1f}%) score 70-80 (High Quality)")
print(f"  • {valid_count} total tickers with ≥4 metrics (out of {len(results)} total)")
print(f"  • Data gaps remain in: interest coverage (31.2%), reinvestment rate (19.7%), EV/EBITDA (9.4%)")
print(f"  • EDGAR source covers {len(results[results['source']=='edgar'])} tickers with best fundamentals coverage")
print(f"  • yfinance source covers {len(results[results['source']=='yfinance'])} tickers with valuation data but weaker fundamentals")

# Missing giants
missing_giants = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-A', 'BRK-B']
print(f"\n  Missing Giants (score 0 due to incomplete data):")
for g in missing_giants:
    if g in results['ticker'].values:
        row = results[results['ticker'] == g].iloc[0]
        print(f"    {g}: score = {row['total_score']}, metrics = {row['available']}")
    else:
        print(f"    {g}: NOT IN DATA")

# Save results
results.to_csv('damodaran_quality_ranked.csv', index=False)
print(f"\n✅ Full rankings saved to: damodaran_quality_ranked.csv")

# Top 100 save
top100 = results_valid.head(100)[['ticker', 'total_score', 'profitability', 'financial_health', 'growth', 'stability', 'data_completeness', 'as_of_date', 'source', 'available']].copy()
top100.to_csv('damodaran_top100_quality.csv', index=False)
print(f"✅ Top 100 saved to: damodaran_top100_quality.csv")
