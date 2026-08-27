#!/usr/bin/env python3
"""
EDGAR Pipeline v2 — Implementation Summary
===========================================

FILES CREATED:
- edgar_lib.py              — Shared extraction library (frame parsing, differencing, provenance)
- edgar_html_10q.py         — HTML 10-Q parser (fallback for incomplete XBRL)
- unified_edgar_pipeline.py — XBRL + HTML merger with smart fallback
- full_universe_backfill.py — Universe-wide additive backfill
- edgar_hygiene_audit.py    — Sample testing and reporting

KEY FIX: PANW Static FCF
  Root cause: EDGAR XBRL reports cash flow as fiscal-year cumulative YTD,
  not standalone quarterly values. N/A frames = fiscal cumulative.
  
  Fix: parse_cashflow_quarterly() differences within fiscal year:
    Q1 = N/A frame (3-month cumulative)
    Q2 = N/A frame (6-month) - Q1
    Q3 = N/A frame (9-month) - Q2
    Q4 = CYyyyy (12-month) - CYyyyyQ3 (9-month)

PROVENANCE TRACKING:
  Every financial value has a _provenance suffix indicating source:
    - "reported":      Direct from XBRL/HTML
    - "computed":      Derived (TTM sum, FY - Q3 diff)
    - "proxy":         Fallback (OCF used as FCF when CapEx missing)
    - "unavailable":   No data found
    - "missing":       Tag not present in companyfacts

HYGIENE AUDIT (29 tickers):
  Quality: 76% Good (70-89), 10% Fair (50-69), 0% Poor
  FCF Sources: 85% computed, 14% proxy, 0.6% unavailable
  HTML fallback used for 4 tickers (JPM, BAC, CVX, CAT)

UNIVERSE BACKFILL (156 monitored tickers):
  Processed: 156 tickers in 4.6 minutes
  OK: 133 (85%), No CIK: 18, No data: 5, Errors: 0
  Total fundamentals rows: 311,631 (8,669 unique tickers)

NEXT STEPS:
  1. Run full backfill (8,669 tickers) — ~25 minutes
  2. Delete stale fundamentals_history_backfill (synthetic noise)
  3. Rebuild preferred_metrics_history
  4. Add earnings_stability computation from quarterly NI trend
"""
