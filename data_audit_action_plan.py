#!/usr/bin/env python3
"""
DATA AUDIT RESULTS & ACTION PLAN
"""

print("=" * 80)
print("DATA AUDIT RESULTS & ACTION PLAN")
print("=" * 80)
print()

print("EXECUTIVE SUMMARY")
print("-" * 80)
print("""
CRITICAL ISSUES FOUND:

1. DAILY PRICES — SEVERITY: HIGH
   - 222 tickers with no prices in last 7 days
   - 750 zero-price rows
   - 106,528 NEGATIVE price rows (data error)
   - Infinite values in adj_close (data error)
   - 95.7% missing market_cap (only 1.4M rows have it)
   
2. FUNDAMENTALS — SEVERITY: HIGH
   - 2,212 tickers stale (no data in 90 days)
   - 142 rows with FUTURE dates (data error)
   - 54.7% missing revenue_quarterly
   - 72.8% missing total_debt
   - 18.6% missing net_income
   
3. MONITORED STOCKS COVERAGE — SEVERITY: MEDIUM
   - PANW shows as missing recent fundamentals (but EDGAR has data!)
   - 7 stocks missing fundamentals entirely
   - 11 stocks missing recent fundamentals
   
4. POST-PROCESSED TABLES — SEVERITY: MEDIUM
   - preferred_metrics_history not refreshed after data fixes
   - Contains same stale/fundamentals issues as source
""")

print("=" * 80)
print("ACTION PLAN")
print("=" * 80)
print()

print("PRIORITY 1 — Fix Data Errors (Daily Prices)")
print("-" * 80)
print("""
a) Remove zero and negative price rows:
   - 750 zero prices
   - 106,528 negative prices
   - These are data errors that corrupt calculations

b) Fix infinite values:
   - Replace inf/-inf with NaN
   - Likely caused by division by zero in returns calculations

c) Backfill missing market_cap:
   - Only 4.3% of rows have market_cap
   - Need to compute: price * shares_outstanding
   - Source from fundamentals.shares_outstanding

d) Remove stale tickers:
   - 222 tickers with no recent prices
   - Either delist or mark as inactive
""")

print()
print("PRIORITY 2 — Fix Fundamentals Issues")
print("-" * 80)
print("""
a) Remove future-dated rows:
   - 142 rows with dates > today
   - Likely data entry errors or timezone issues

b) Improve revenue coverage:
   - Only 45.3% have revenue_quarterly
   - Parse additional XBRL tags for revenue
   - Use HTML 10-Q fallback for missing filers

c) Fix PANW coverage:
   - PANW has EDGAR data through 2026-04-30
   - Our fundamentals shows it as missing recent data
   - Issue: fiscal year misalignment in date comparison
   - Fix: normalize fiscal quarter dates before comparison

d) Backfill missing total_debt:
   - Only 27.2% have total_debt
   - Parse from balance sheet XBRL tags
   - Critical for debt_to_equity calculation
""")

print()
print("PRIORITY 3 — Refresh Post-Processed Tables")
print("-" * 80)
print("""
a) preferred_metrics_history:
   - Rebuild after fundamentals cleanup
   - Filter to recent data only (last 5 years)
   - Ensure all monitored stocks are included

b) Recalculate derived metrics:
   - ROE, ROIC, D/E need refresh
   - Use corrected debt and equity values
   - Add provenance tracking for each metric

c) Damodaran quality scores:
   - Re-run quality scoring after data fixes
   - Update top-100 rankings
   - Verify PANW and other key names are scored correctly
""")

print()
print("PRIORITY 4 — Monitored Stocks Coverage")
print("-" * 80)
print("""
a) Fix missing price data:
   - CYBR has no price data
   - Verify CIK mapping and ticker symbol

b) Fix missing fundamentals:
   - ARKK, CYBR, QQQ, TEST, VNQ, VUG, XBI
   - These are ETFs — may not have EDGAR data
   - Use ETF-specific data sources

c) Fix PANW recent data:
   - PANW has EDGAR_v2 data through 2026-04-30
   - Date comparison logic needs fixing
   - Verify fiscal quarter alignment
""")

print()
print("=" * 80)
print("VERIFICATION STEPS")
print("=" * 80)
print("""
After fixes, verify:
1. Zero/negative prices removed
2. Future dates removed from fundamentals
3. PANW recent quarters appear in fundamentals
4. preferred_metrics_history refreshed
5. All monitored stocks have price + fundamentals data
6. Damodaran quality scores recalculated
""")