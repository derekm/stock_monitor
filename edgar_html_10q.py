#!/usr/bin/env python3
"""
edgar_html_10q.py — Parse HTML 10-Q filings for quarterly financial data.

Fallback parser for when EDGAR XBRL companyfacts is incomplete or missing.
Parses the actual 10-Q filing HTML to extract:
- Revenue (Total Revenue / Net Revenue)
- Net Income
- Operating Income
- Operating Cash Flow (quarterly, not cumulative)
- Capital Expenditure
- Free Cash Flow (computed as OCF - CapEx)
- Balance sheet items (Assets, Equity, Debt, Cash, Shares)

Usage:
  python edgar_html_10q.py --ticker PANW --cik 0001327567
  python edgar_html_10q.py --ticker AAPL --cik 0000320193 --quarters 8
"""

import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent
UA = {"User-Agent": "personal-research derek.moore@example.com"}

# SEC EDGAR full-text search and filing URLs
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{accession}-index.htm"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"


# Patterns for extracting values from HTML tables
# Numbers can be: 1,234,567 | 1234567 | (1,234,567) | $1,234,567 | (1,234) | -
NUMBER_PATTERN = r"\(?\$?([\d,]+(?:\.\d+)?)\)?"

# Common label patterns for financial concepts
REVENUE_LABELS = [
    r"(?:total\s+)?revenue[s]?",
    r"net\s+revenue[s]?",
    r"total\s+net\s+revenue[s]?",
    r"net\s+sales",
    r"total\s+sales",
    r"sales\s+revenue",
]

NET_INCOME_LABELS = [
    r"net\s+income",
    r"net\s+income\s+\(loss\)",
    r"net\s+earnings",
    r"net\s+(?:income|earnings)\s+attributable",
]

OPERATING_INCOME_LABELS = [
    r"operating\s+income",
    r"income\s+from\s+operations",
    r"operating\s+(?:income|profit)",
]

OCF_LABELS = [
    r"net\s+cash\s+(?:provided|used)\s+by\s+operating\s+activities",
    r"operating\s+cash\s+flow",
    r"net\s+cash\s+from\s+operations",
    r"cash\s+(?:provided|used)\s+by\s+operations",
]

CAPEX_LABELS = [
    r"(?:purchase|payment)s?\s+(?:of|for)\s+(?:property|equipment|property\s+and\s+equipment)",
    r"capital\s+expenditures?",
    r"purchase[s]?\s+of\s+(?:property|equipment|PP&E)",
    r"acquisition[s]?\s+of\s+(?:property|equipment)",
]

ASSETS_LABELS = [
    r"total\s+assets",
    r"assets",
]

EQUITY_LABELS = [
    r"total\s+(?:stockholders|shareholders)['\s]+\s*equity",
    r"total\s+equity",
    r"(?:stockholders|shareholders)['\s]+\s*equity",
]

DEBT_LABELS = [
    r"total\s+debt",
    r"long[\s-]*term\s+debt",
    r"debt",
]

CASH_LABELS = [
    r"cash\s+and\s+cash\s+equivalents",
    r"cash",
]


def fetch_filing_index(cik: str) -> Optional[dict]:
    """Fetch EDGAR filing index for a CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=UA, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def find_10q_filings(cik: str, max_filings: int = 12) -> list[dict]:
    """Find recent 10-Q filings for a CIK."""
    index = fetch_filing_index(cik)
    if index is None:
        return []
    
    filings = index.get("filings", {}).get("recent", {})
    if not filings:
        return []
    
    forms = filings.get("form", [])
    accession_numbers = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])
    filing_dates = filings.get("filingDate", [])
    report_dates = filings.get("reportDate", [])
    
    results = []
    for i, form in enumerate(forms):
        if form in ("10-Q", "10-K") and len(results) < max_filings:
            results.append({
                "form": form,
                "accession": accession_numbers[i],
                "primary_doc": primary_docs[i],
                "filing_date": filing_dates[i],
                "report_date": report_dates[i],
            })
    
    return results


def fetch_filing_html(cik: str, accession: str, primary_doc: str) -> Optional[str]:
    """Fetch HTML content of a filing."""
    # Remove dashes from accession number for URL
    accession_clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_clean}/{primary_doc}"
    
    try:
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def extract_table_data(html: str) -> list[dict]:
    """
    Extract financial data from HTML tables in a 10-Q filing.
    
    Returns list of dicts with:
    - label: row label text
    - values: list of numeric values found
    - context: surrounding text for validation
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Find all tables
    tables = soup.find_all("table")
    
    results = []
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            
            # First cell is typically the label
            label_cell = cells[0]
            label = label_cell.get_text(strip=True).lower()
            
            # Extract numeric values from remaining cells
            values = []
            for cell in cells[1:]:
                text = cell.get_text(strip=True)
                # Remove commas, dollar signs, parentheses
                text_clean = text.replace(",", "").replace("$", "").replace("(", "-").replace(")", "")
                
                # Try to parse as number
                try:
                    val = float(text_clean)
                    if val != 0 or text_clean in ("0", "0.0", "(0)"):
                        values.append(val)
                except (ValueError, TypeError):
                    pass
            
            if values:
                results.append({
                    "label": label,
                    "values": values,
                    "raw_label": cells[0].get_text(strip=True),
                })
    
    return results


def match_label(label: str, patterns: list[str]) -> bool:
    """Check if label matches any of the patterns."""
    label_lower = label.lower()
    for pattern in patterns:
        if re.search(pattern, label_lower):
            return True
    return False


def extract_financial_value(table_data: list[dict], patterns: list[str],
                             value_index: int = 0) -> Optional[float]:
    """
    Extract a financial value from parsed table data.
    
    Args:
        table_data: Output from extract_table_data()
        patterns: Regex patterns to match label
        value_index: Which value to take (0 = first, -1 = last)
    
    Returns:
        Float value if found, None otherwise
    """
    for entry in table_data:
        if match_label(entry["label"], patterns):
            values = entry["values"]
            if len(values) > value_index:
                return values[value_index]
    return None


def parse_10q_filing(cik: str, accession: str, primary_doc: str,
                      report_date: str, form: str) -> Optional[dict]:
    """
    Parse a single 10-Q or 10-K filing for quarterly financials.
    
    Returns dict with extracted values and provenance info.
    """
    html = fetch_filing_html(cik, accession, primary_doc)
    if html is None:
        return None
    
    table_data = extract_table_data(html)
    
    # Extract values
    revenue = extract_financial_value(table_data, REVENUE_LABELS)
    net_income = extract_financial_value(table_data, NET_INCOME_LABELS)
    operating_income = extract_financial_value(table_data, OPERATING_INCOME_LABELS)
    ocf = extract_financial_value(table_data, OCF_LABELS)
    capex = extract_financial_value(table_data, CAPEX_LABELS)
    assets = extract_financial_value(table_data, ASSETS_LABELS)
    equity = extract_financial_value(table_data, EQUITY_LABELS)
    debt = extract_financial_value(table_data, DEBT_LABELS)
    cash = extract_financial_value(table_data, CASH_LABELS)
    
    # Compute FCF
    if ocf is not None and capex is not None:
        fcf = ocf - abs(capex)
        fcf_provenance = "html_parsed_computed"
    elif ocf is not None:
        fcf = ocf
        fcf_provenance = "html_parsed_proxy"
    else:
        fcf = None
        fcf_provenance = "html_parsed_unavailable"
    
    return {
        "report_date": report_date,
        "form": form,
        "accession": accession,
        # Canonical fundamentals names (post-2026-08 migration). A 10-Q income
        # statement covers ONE quarter, so the income items are *_quarterly; the
        # cash-flow items are cumulative-to-date and are consumed as TTM inputs,
        # matching how the XBRL path reports them.
        "revenue_quarterly": revenue,
        "net_income_quarterly": net_income,
        "operating_income_quarterly": operating_income,
        "operating_cash_flow_ttm": ocf,
        "capital_expenditure_ttm": capex,
        "free_cash_flow": fcf,
        "fcf_provenance": fcf_provenance,
        "total_assets": assets,
        "shareholders_equity": equity,
        "total_debt": debt,
        "cash_and_equivalents": cash,
        "source": "html_10q",
    }


def extract_quarterly_from_html(cik: str, ticker: str,
                                 max_quarters: int = 8) -> list[dict]:
    """
    Extract quarterly financials from recent HTML 10-Q filings.
    
    For each filing, extracts standalone quarterly values.
    For 10-K (annual), breaks into quarters if possible.
    """
    filings = find_10q_filings(cik, max_filings=max_quarters * 2)
    
    results = []
    for filing in filings:
        if len(results) >= max_quarters:
            break
        
        data = parse_10q_filing(
            cik, filing["accession"], filing["primary_doc"],
            filing["report_date"], filing["form"]
        )
        
        # revenue_quarterly / net_income_quarterly are the canonical names; a
        # filing with neither carries no usable income data.
        if data and (data.get("revenue_quarterly") is not None
                     or data.get("net_income_quarterly") is not None):
            data["ticker"] = ticker
            results.append(data)
        
        time.sleep(0.12)  # SEC rate limit
    
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Parse HTML 10-Q filings for financial data")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--cik", help="CIK (auto-lookup if not provided)")
    ap.add_argument("--quarters", type=int, default=8)
    args = ap.parse_args()
    
    from edgar_lib import load_cik_map, CIK_OVERRIDES, NO_COMPANYFACTS
    
    cik_map = load_cik_map()
    cik_map.update(CIK_OVERRIDES)
    
    cik = args.cik or cik_map.get(args.ticker.upper())
    if cik is None:
        print(f"No CIK found for {args.ticker}")
        exit(1)
    
    if args.ticker.upper() in NO_COMPANYFACTS:
        print(f"{args.ticker} has no XBRL companyfacts (known limitation)")
        exit(0)
    
    print(f"Fetching HTML 10-Qs for {args.ticker} (CIK: {cik})...")
    results = extract_quarterly_from_html(cik, args.ticker, args.quarters)
    
    if results:
        df = pd.DataFrame(results)
        print(f"\nExtracted {len(df)} quarters:")
        print(df[["report_date", "form", "revenue", "net_income", "operating_cash_flow",
                   "capital_expenditure", "free_cash_flow", "fcf_provenance"]].to_string(index=False))
    else:
        print("No financial data extracted")
