#!/usr/bin/env python3
"""
unified_edgar_pipeline.py — Combines XBRL + HTML parsing with smart fallback.

Priority order:
1. EDGAR XBRL (edgar_lib.py) — fastest, most reliable for most filers
2. HTML 10-Q parsing (edgar_html_10q.py) — fallback for incomplete XBRL

Merges results: if XBRL FCF is "proxy" or "unavailable", prefer HTML parsed value.

Usage:
  python unified_edgar_pipeline.py --ticker PANW
  python unified_edgar_pipeline.py --max-tickers 50
  python unified_edgar_pipeline.py --full-universe
"""

import time
from pathlib import Path
from typing import Optional

import pandas as pd

from edgar_lib import (
    load_cik_map, get_cik, CIK_OVERRIDES, NO_COMPANYFACTS,
    extract_financials, compute_quarterly_fundamentals, detect_fiscal_year_end,
    parse_quarterly, parse_cashflow_quarterly, parse_balance, TAG_MAP,
    fetch_companyfacts, extract_facts
)
from edgar_html_10q import extract_quarterly_from_html


def unified_extract(cik: str, ticker: str, use_html: bool = True,
                     max_html_quarters: int = 4) -> dict:
    """
    Unified extraction combining XBRL and HTML data sources.
    
    Returns dict with:
    - rows: list of quarterly fundamental dicts
    - xbrl_count: number of rows from XBRL
    - html_count: number of rows from HTML
    - merged_count: number of merged rows
    - fcf_provenance_dist: distribution of FCF sources
    - quality_score: 0-100 data quality score
    """
    result = {
        "rows": [],
        "xbrl_count": 0,
        "html_count": 0,
        "merged_count": 0,
        "fcf_provenance_dist": {},
        "quality_score": 0,
    }
    
    # Step 1: Extract from XBRL
    xbrl_rows = []
    try:
        fin = extract_financials(cik)
        if fin:
            xbrl_rows = compute_quarterly_fundamentals(fin, ticker)
    except Exception as e:
        print(f"  XBRL extraction failed for {ticker}: {e}")
    
    result["xbrl_count"] = len(xbrl_rows)
    
    # Step 2: Check if we need HTML fallback
    # Conditions: no XBR rows, or all FCF is proxy/unavailable
    need_html = use_html and (
        len(xbrl_rows) == 0 or
        all(r.get("fcf_provenance") in ("proxy", "unavailable") for r in xbrl_rows)
    )
    
    html_rows = []
    if need_html:
        try:
            html_rows = extract_quarterly_from_html(cik, ticker, max_html_quarters)
        except Exception as e:
            print(f"  HTML extraction failed for {ticker}: {e}")
        
        result["html_count"] = len(html_rows)
    
    # Step 3: Merge results
    # Build lookup by date (normalize all to date objects)
    def _normalize_date(d):
        if isinstance(d, str):
            return pd.Timestamp(d).date()
        if hasattr(d, 'date'):
            return d.date() if not isinstance(d.date(), type) else d  # already a date
        return d
    
    xbrl_by_date = {_normalize_date(r["as_of_date"]): r for r in xbrl_rows}
    html_by_date = {_normalize_date(r.get("report_date")): r for r in html_rows if r.get("report_date")}
    
    # For each date, prefer HTML if XBRL is missing or proxy
    merged = []
    all_dates = sorted(set(xbrl_by_date.keys()) | set(html_by_date.keys()))
    
    for date_key in all_dates:
        xbrl_row = xbrl_by_date.get(date_key)
        html_row = html_by_date.get(date_key)
        
        if xbrl_row and html_row:
            # Merge: prefer HTML for FCF if XBRL is proxy
            merged_row = xbrl_row.copy()
            if xbrl_row.get("fcf_provenance") in ("proxy", "unavailable"):
                if html_row.get("free_cash_flow") is not None:
                    merged_row["free_cash_flow"] = html_row["free_cash_flow"]
                    merged_row["fcf_provenance"] = html_row.get("fcf_provenance", "html_merged")
                    merged_row["capital_expenditure"] = html_row.get("capital_expenditure")
            merged.append(merged_row)
        elif xbrl_row:
            merged.append(xbrl_row)
        elif html_row:
            # Convert HTML row to standard format
            merged.append({
                "ticker": ticker,
                "as_of_date": html_row.get("report_date"),
                "total_revenue": html_row.get("revenue"),
                "net_income": html_row.get("net_income"),
                "ebit": html_row.get("operating_income"),
                "free_cash_flow": html_row.get("free_cash_flow"),
                "fcf_provenance": html_row.get("fcf_provenance"),
                "capital_expenditure": html_row.get("capital_expenditure"),
                "total_assets": html_row.get("assets"),
                "shareholders_equity": html_row.get("equity"),
                "total_debt": html_row.get("debt"),
                "cash_and_equivalents": html_row.get("cash"),
                "source": "html_10q",
            })
    
    result["merged_count"] = len(merged)
    
    # FCF provenance distribution
    provenance_counts = {}
    for r in merged:
        prov = r.get("fcf_provenance", "unknown")
        provenance_counts[prov] = provenance_counts.get(prov, 0) + 1
    result["fcf_provenance_dist"] = provenance_counts
    
    # Quality score
    if merged:
        total = len(merged)
        has_revenue = sum(1 for r in merged if r.get("total_revenue") is not None)
        has_fcf = sum(1 for r in merged if r.get("free_cash_flow") is not None)
        has_capex = sum(1 for r in merged if r.get("capital_expenditure") is not None)
        has_ocf = sum(1 for r in merged if r.get("ttm_operating_cash_flow") is not None)
        
        result["quality_score"] = int((
            has_revenue / total * 25 +
            has_ocf / total * 25 +
            has_fcf / total * 25 +
            has_capex / total * 25
        ))
    
    result["rows"] = merged
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--max-tickers", type=int, default=10)
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()
    
    cik_map = load_cik_map()
    cik_map.update(CIK_OVERRIDES)
    
    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        # Use monitored stocks
        MONITORED = Path("monitored_stocks.parquet")
        if MONITORED.exists():
            stocks = pd.read_parquet(MONITORED)
            tickers = sorted(stocks["ticker"].astype(str).str.upper().unique().tolist())
        else:
            tickers = sorted(cik_map.keys())
        
        tickers = [t for t in tickers if t not in NO_COMPANYFACTS]
        tickers = tickers[: args.max_tickers]
    
    print(f"Unified extraction for {len(tickers)} tickers...")
    
    all_results = []
    for t in tickers:
        cik = get_cik(t, cik_map)
        if cik is None:
            print(f"  !! {t}: no CIK")
            continue
        
        print(f"  {t}...", end=" ")
        result = unified_extract(cik, t, use_html=not args.no_html)
        
        xbrl = result["xbrl_count"]
        html = result["html_count"]
        merged = result["merged_count"]
        quality = result["quality_score"]
        provenance = result["fcf_provenance_dist"]
        
        print(f"XBRL={xbrl}, HTML={html}, Merged={merged}, Quality={quality}, FCF sources={provenance}")
        
        all_results.extend(result["rows"])
        
        time.sleep(0.12)
    
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\nTotal rows: {len(df)}")
        print(df[["ticker", "as_of_date", "total_revenue", "free_cash_flow", "fcf_provenance", "source"]].head(20).to_string())
