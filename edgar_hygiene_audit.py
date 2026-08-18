#!/usr/bin/env python3
"""
edgar_hygiene_audit.py — Run unified extraction on a diverse sample and produce
hygiene report covering:
- Data coverage by source (XBRL vs HTML)
- FCF provenance distribution
- Fiscal year detection accuracy
- Known-good data validation
- Missing/malformed data identification
"""

import time
from pathlib import Path

import pandas as pd

from edgar_lib import load_cik_map, CIK_OVERRIDES, NO_COMPANYFACTS, get_cik
from unified_edgar_pipeline import unified_extract


# Diverse sample covering various fiscal year ends and filer types
SAMPLE_TICKERS = [
    # Non-December fiscal years
    ("PANW", 7),   # July
    ("AAPL", 9),   # September
    ("MSFT", 6),   # June
    ("NVDA", 0),   # January (approx)
    ("WMT", 0),    # January
    ("NFLX", 11),  # December (calendar)
    ("GOOGL", 11), # December
    ("AMZN", 11),  # December
    ("META", 11),  # December
    ("TSLA", 11),  # December
    
    # Mid-cap and small-cap
    ("ADMA", 11),  # December
    ("CRWD", 0),   # January
    ("ZS", 6),     # July
    ("OKTA", 0),   # January
    ("TWTR", 11),  # December
    
    # International/ADR
    ("BABA", 2),   # March
    ("TSM", 11),   # December
    ("SAP", 11),   # December
    
    # Financials (different reporting)
    ("BRK-B", 11), # December
    ("JPM", 11),   # December
    ("BAC", 11),   # December
    
    # Healthcare
    ("JNJ", 11),   # December
    ("UNH", 11),   # December
    ("PFE", 11),   # December
    ("ABBV", 11),  # December
    
    # Industrial/Other
    ("BA", 11),    # December
    ("CAT", 11),   # December
    ("GE", 11),    # December
    
    # Energy
    ("XOM", 11),   # December
    ("CVX", 11),   # December
]


def run_hygiene_audit():
    """Run unified extraction on sample and produce hygiene report."""
    cik_map = load_cik_map()
    cik_map.update(CIK_OVERRIDES)
    
    results = []
    all_rows = []
    
    print("=" * 80)
    print("EDGAR HYGIENE AUDIT — Sample Extraction")
    print("=" * 80)
    
    for ticker, expected_fye in SAMPLE_TICKERS:
        cik = get_cik(ticker, cik_map)
        if cik is None:
            results.append({
                "ticker": ticker,
                "status": "no_cik",
                "xbrl_count": 0,
                "html_count": 0,
                "merged_count": 0,
                "quality": 0,
            })
            continue
        
        try:
            result = unified_extract(cik, ticker, use_html=True, max_html_quarters=4)
            
            rows = result["rows"]
            
            # Check for common issues
            issues = []
            if result["xbrl_count"] == 0:
                issues.append("no_xbrl_data")
            if result["html_count"] > 0 and result["xbrl_count"] > 0:
                issues.append("html_fallback_used")
            
            # Check FCF provenance
            fcf_prov = result["fcf_provenance_dist"]
            if fcf_prov.get("unavailable", 0) > 0:
                issues.append("some_fcf_unavailable")
            if fcf_prov.get("proxy", 0) > fcf_prov.get("computed", 0):
                issues.append("fcf_mostly_proxy")
            
            # Check for data completeness
            if rows:
                has_rev = sum(1 for r in rows if r.get("revenue_quarterly") is not None)
                has_fcf = sum(1 for r in rows if r.get("free_cash_flow") is not None)
                has_ocf = sum(1 for r in rows if r.get("operating_cash_flow_ttm") is not None)
                
                if has_rev / len(rows) < 0.5:
                    issues.append("revenue_sparse")
                if has_fcf / len(rows) < 0.5:
                    issues.append("fcf_sparse")
                if has_ocf / len(rows) < 0.5:
                    issues.append("ocf_sparse")
            
            results.append({
                "ticker": ticker,
                "status": "ok" if not issues else "issues",
                "issues": ";".join(issues) if issues else None,
                "xbrl_count": result["xbrl_count"],
                "html_count": result["html_count"],
                "merged_count": result["merged_count"],
                "quality": result["quality_score"],
                "fcf_computed": fcf_prov.get("computed", 0),
                "fcf_proxy": fcf_prov.get("proxy", 0),
                "fcf_unavailable": fcf_prov.get("unavailable", 0),
            })
            
            all_rows.extend(rows)
            
            time.sleep(0.12)
        except Exception as e:
            results.append({
                "ticker": ticker,
                "status": "error",
                "error": str(e),
                "xbrl_count": 0,
                "html_count": 0,
                "merged_count": 0,
                "quality": 0,
            })
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("HYGIENE REPORT SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal tickers tested: {len(SAMPLE_TICKERS)}")
    print(f"Successful extractions: {(df['status'] != 'error').sum()}")
    print(f"No CIK: {(df['status'] == 'no_cik').sum()}")
    print(f"With issues: {(df['status'] == 'issues').sum()}")
    print(f"Errors: {(df['status'] == 'error').sum()}")
    
    print(f"\nQuality Score Distribution:")
    print(f"  90-100 (Excellent): {(df['quality'] >= 90).sum()}")
    print(f"  70-89  (Good):      {((df['quality'] >= 70) & (df['quality'] < 90)).sum()}")
    print(f"  50-69  (Fair):      {((df['quality'] >= 50) & (df['quality'] < 70)).sum()}")
    print(f"  0-49   (Poor):      {(df['quality'] < 50).sum()}")
    
    print(f"\nData Sources:")
    print(f"  XBRL rows: {df['xbrl_count'].sum()}")
    print(f"  HTML rows: {df['html_count'].sum()}")
    print(f"  Merged rows: {df['merged_count'].sum()}")
    
    print(f"\nFCF Provenance:")
    print(f"  Computed (OCF - CapEx): {df['fcf_computed'].sum()}")
    print(f"  Proxy (OCF only):       {df['fcf_proxy'].sum()}")
    print(f"  Unavailable:            {df['fcf_unavailable'].sum()}")
    
    print(f"\nIssue Distribution:")
    all_issues = []
    for issues_str in df["issues"].dropna():
        all_issues.extend(issues_str.split(";"))
    
    if all_issues:
        from collections import Counter
        issue_counts = Counter(all_issues)
        for issue, count in issue_counts.most_common():
            print(f"  {issue}: {count}")
    
    print(f"\nPer-Ticker Detail:")
    print("-" * 80)
    for _, row in df.iterrows():
        status_icon = "✓" if row["status"] == "ok" else "⚠" if row["status"] == "issues" else "✗"
        issues_str = f" ({row.get('issues', '')})" if pd.notna(row.get('issues')) else ""
        print(f"  {status_icon} {row['ticker']:<8} XBRL={row['xbrl_count']:3d}, HTML={row['html_count']:2d}, "
              f"Merged={row['merged_count']:3d}, Quality={row['quality']:3d}{issues_str}")
    
    if all_rows:
        print(f"\n" + "=" * 80)
        print("DATA QUALITY SAMPLES")
        print("=" * 80)
        
        rows_df = pd.DataFrame(all_rows)
        
        # Show sample rows with full provenance
        print("\nSample PANW rows (most recent 4):")
        panw = rows_df[rows_df["ticker"] == "PANW"].tail(4)
        for _, row in panw.iterrows():
            print(f"  {row.get('as_of_date')}: Rev={row.get('revenue_quarterly')}, "
                  f"TTM_OCF={row.get('operating_cash_flow_ttm')}, "
                  f"TTM_CapEx={row.get('capital_expenditure_ttm')}, "
                  f"FCF={row.get('free_cash_flow')} ({row.get('fcf_provenance')})")
        
        print("\nSample AAPL rows (most recent 4):")
        aapl = rows_df[rows_df["ticker"] == "AAPL"].tail(4)
        for _, row in aapl.iterrows():
            print(f"  {row.get('as_of_date')}: Rev={row.get('revenue_quarterly')}, "
                  f"TTM_OCF={row.get('operating_cash_flow_ttm')}, "
                  f"TTM_CapEx={row.get('capital_expenditure_ttm')}, "
                  f"FCF={row.get('free_cash_flow')} ({row.get('fcf_provenance')})")
    
    return df, all_rows


if __name__ == "__main__":
    results_df, rows = run_hygiene_audit()
    
    # Save results
    results_df.to_csv("edgar_hygiene_audit_summary.csv", index=False)
    pd.DataFrame(rows).to_csv("edgar_hygiene_audit_rows.csv", index=False)
    print("\n✓ Results saved to edgar_hygiene_audit_summary.csv")
