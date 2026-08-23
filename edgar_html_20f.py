#!/usr/bin/env python3
"""
edgar_html_20f.py — Parse HTML 20-F/40-F filings for annual financial data.

Fallback parser for foreign private issuers when EDGAR XBRL companyfacts is
incomplete or missing. Parses the actual 20-F/40-F filing HTML to extract:
- Revenue (Total Revenue / Net Revenue)
- Net Income
- Operating Income
- Operating Cash Flow (annual)
- Capital Expenditure
- Free Cash Flow (computed as OCF - CapEx)
- Balance sheet items (Assets, Equity, Debt, Cash, Shares)

Usage:
  python edgar_html_20f.py --ticker BAYRY --cik 00008763 --accession 0001193125-24-000001
  python edgar_html_20f.py --ticker B --cik 0001069183 --accession 0001069183-25-000001
"""

import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent
UA = {"User-Agent": "personal-research derek.moore@example.com"}

# SEC EDGAR filing URLs
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{accession}-index.htm"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"

# Number pattern: 1,234,567 | 1234567 | (1,234,567) | $1,234,567 | (1,234) | -
NUMBER_PATTERN = r"\(?\$?([\d,]+(?:\.\d+)?)\)?"

# Common label patterns for financial concepts in 20-F/40-F (IFRS terminology)
# IFRS uses different labels than US-GAAP
REVENUE_LABELS = [
    r"(?:total\s+)?revenue[s]?",
    r"net\s+revenue[s]?",
    r"total\s+net\s+revenue[s]?",
    r"net\s+sales",
    r"total\s+sales",
    r"sales\s+revenue",
    r"turnover",                    # IFRS common
    r"revenue\s+from\s+contracts",
]

NET_INCOME_LABELS = [
    r"net\s+income",
    r"net\s+income\s+\(loss\)",
    r"net\s+earnings",
    r"profit\s+for\s+the\s+(?:year|period)",
    r"profit\s+\(loss\)\s+for\s+the\s+(?:year|period)",
    r"net\s+profit",
    r"comprehensive\s+income",      # IFRS
]

OPERATING_INCOME_LABELS = [
    r"operating\s+income",
    r"income\s+from\s+operations",
    r"operating\s+(?:income|profit)",
    r"operating\s+profit",
    r"earnings\s+before\s+interest\s+and\s+tax",  # EBIT
]

OCF_LABELS = [
    r"net\s+cash\s+(?:provided|used)\s+by\s+operating\s+activities",
    r"operating\s+cash\s+flow",
    r"net\s+cash\s+from\s+operations",
    r"cash\s+(?:provided|used)\s+by\s+operations",
    r"cash\s+flows?\s+from\s+operating\s+activities",
]

CAPEX_LABELS = [
    r"(?:purchase|payment)s?\s+(?:of|for)\s+(?:property|equipment|property\s+and\s+equipment)",
    r"capital\s+expenditures?",
    r"purchase[s]?\s+of\s+(?:property|equipment|PP&E)",
    r"acquisition[s]?\s+of\s+(?:property|equipment)",
    r"additions\s+to\s+property",
]

ASSETS_LABELS = [
    r"^total\s+assets$",
    r"total\s+assets\s+\(.*\)",
]

EQUITY_LABELS = [
    r"total\s+equity",
    r"shareholders?'\s+equity",
    r"stockholders?'\s+equity",
    r"equity\s+attributable\s+to\s+owners",
    r"total\s+shareholders?'\s+equity",
]

DEBT_LABELS = [
    r"^total\s+debt$",
    r"total\s+debt\s+and\s+finance\s+lease",
    r"long.term\s+debt",
    r"non.current\s+borrowings",
    r"loans\s+and\s+borrowings",
]

CASH_LABELS = [
    r"^cash\s+and\s+cash\s+equivalents$",
    r"cash\s+and\s+short.term\s+investments",
]

SHARES_LABELS = [
    r"weighted.average\s+(?:number\s+of\s+)?shares",
    r"basic\s+(?:and\s+diluted\s+)?(?:earnings\s+per\s+share|EPS)",
    r"outstanding\s+shares",
    r"issued\s+shares",
    r"number\s+of\s+shares\s+outstanding",
]


def parse_number(text: str) -> Optional[float]:
    """Parse a financial number from text, handling (negative), commas, $."""
    if not text or text.strip() in ("-", "—", "", "N/A", "n/a"):
        return None
    text = text.strip().replace(",", "")
    # Handle parentheses as negative
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    text = text.replace("$", "")
    try:
        return float(text)
    except ValueError:
        return None


def find_table_rows(soup: BeautifulSoup, labels: List[str]) -> List[Dict[str, Any]]:
    """Find table rows matching any of the label patterns."""
    matches = []
    for label_pattern in labels:
        # Search in th, td, and text nodes
        for elem in soup.find_all(text=re.compile(label_pattern, re.I)):
            row = elem.find_parent("tr")
            if row:
                cells = row.find_all(["td", "th"])
                cell_texts = [c.get_text(strip=True) for c in cells]
                # First cell is usually the label
                label = cell_texts[0] if cell_texts else ""
                # Remaining cells are values (often multiple periods)
                values = cell_texts[1:] if len(cell_texts) > 1 else []
                matches.append({
                    "pattern": label_pattern,
                    "label": label,
                    "values": values,
                    "raw_cells": cell_texts,
                })
    return matches


def extract_concept(soup: BeautifulSoup, labels: List[str], prefer_recent: bool = True) -> Optional[float]:
    """Extract a single financial concept from the filing."""
    matches = find_table_rows(soup, labels)
    if not matches:
        return None
    
    # Prefer the first match with parseable values
    for m in matches:
        for val_text in m["values"]:
            parsed = parse_number(val_text)
            if parsed is not None:
                return parsed
    
    return None


def extract_period_values(soup: BeautifulSoup, labels: List[str], num_periods: int = 3) -> List[Optional[float]]:
    """Extract multiple period values for a concept (current year, prior year, etc.)."""
    matches = find_table_rows(soup, labels)
    if not matches:
        return [None] * num_periods
    
    for m in matches:
        vals = []
        for val_text in m["values"][:num_periods]:
            parsed = parse_number(val_text)
            vals.append(parsed)
        if any(v is not None for v in vals):
            return vals
    
    return [None] * num_periods


def find_filing_documents(index_html: str, form_type: str = "20-F") -> List[Dict[str, str]]:
    """Parse the filing index page to find primary document URLs."""
    soup = BeautifulSoup(index_html, "html.parser")
    docs = []
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        typ = cells[3].get_text(strip=True) if len(cells) > 3 else ""
        desc = cells[1].get_text(strip=True)
        if form_type not in typ and form_type not in desc:
            continue
        link = cells[2].find("a")
        if not link or not link.get("href"):
            continue
        href = link["href"]
        # iXBRL viewer (/ix?doc=...) → raw Archives htm
        if "doc=" in href:
            href = href.split("doc=", 1)[1]
        if not href.startswith("http"):
            href = "https://www.sec.gov" + href
        name = link.get_text(strip=True).split()[0]
        docs.append({
            "form": typ or form_type,
            "description": desc,
            "url": href,
            "filename": name,
        })
    return docs


def _accession_parts(accession: str) -> tuple[str, str]:
    """Return (nodash, dashed 10-2-6) accession."""
    nodash = accession.replace("-", "")
    if len(nodash) >= 18:
        dashed = f"{nodash[:10]}-{nodash[10:12]}-{nodash[12:]}"
    else:
        dashed = accession
    return nodash, dashed


def fetch_filing_html(cik: str, accession: str, primary_doc: str = None) -> Optional[str]:
    """Fetch the 20-F/40-F primary document (not the 30MB+ exhibit dump)."""
    cik_n = str(int(cik))
    nodash, dashed = _accession_parts(accession)
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_n}/{nodash}/{dashed}-index.html"
    r = requests.get(index_url, headers=UA, timeout=60)
    if r.status_code != 200:
        r = requests.get(index_url.replace(".html", ".htm"), headers=UA, timeout=60)
    if r.status_code != 200:
        return None
    if not primary_doc:
        docs = find_filing_documents(r.text, "20-F") or find_filing_documents(r.text, "40-F")
        if not docs:
            return None
        docs = sorted(docs, key=lambda d: (
            0 if "20-f" in d["filename"].lower() or "20f" in d["filename"].lower() else 1,
            0 if d["filename"].lower().endswith((".htm", ".html")) else 1,
            len(d["filename"]),
        ))
        primary_doc = docs[0]["filename"]
        url = docs[0]["url"]
    else:
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_n}/{nodash}/{primary_doc}"
    r = requests.get(url, headers=UA, timeout=120)
    if r.status_code == 200 and r.text:
        return r.text
    return None


def parse_20f_filing(cik: str, accession: str, primary_doc: str = None) -> Dict[str, Any]:
    """Parse a 20-F/40-F filing and extract key financial metrics."""
    html = fetch_filing_html(cik, accession, primary_doc)
    if not html:
        return {"error": "Failed to fetch filing"}
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract financial concepts for multiple periods (current, prior, prior-1)
    results = {
        "cik": cik,
        "accession": accession,
        "filing_type": "20-F/40-F",
        "revenue": extract_period_values(soup, REVENUE_LABELS, 3),
        "net_income": extract_period_values(soup, NET_INCOME_LABELS, 3),
        "operating_income": extract_period_values(soup, OPERATING_INCOME_LABELS, 3),
        "operating_cash_flow": extract_period_values(soup, OCF_LABELS, 3),
        "capex": extract_period_values(soup, CAPEX_LABELS, 3),
        "total_assets": extract_period_values(soup, ASSETS_LABELS, 3),
        "total_equity": extract_period_values(soup, EQUITY_LABELS, 3),
        "total_debt": extract_period_values(soup, DEBT_LABELS, 3),
        "cash": extract_period_values(soup, CASH_LABELS, 3),
        "shares_outstanding": extract_period_values(soup, SHARES_LABELS, 3),
    }
    
    # Compute FCF
    ocf = results["operating_cash_flow"]
    capex = results["capex"]
    fcf = []
    for o, c in zip(ocf, capex):
        if o is not None and c is not None:
            fcf.append(o - c)
        else:
            fcf.append(None)
    results["free_cash_flow"] = fcf
    
    # Try to extract period end dates from the filing
    dates = extract_filing_dates(soup)
    if dates:
        results["period_ends"] = dates[:3]
    
    return results


def extract_filing_dates(soup: BeautifulSoup) -> List[str]:
    """Extract period end dates from the filing header/tables."""
    # Look for date patterns in table headers
    date_pattern = r"(?:December|January|February|March|April|May|June|July|August|September|October|November)\s+\d{1,2},?\s+\d{4}"
    dates = []
    for elem in soup.find_all(text=re.compile(date_pattern)):
        matches = re.findall(date_pattern, elem)
        dates.extend(matches)
    # Deduplicate, keep first 3
    seen = set()
    unique = []
    for d in dates:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique[:3]


def get_latest_20f_accession(cik: str) -> Optional[str]:
    """Get the latest 20-F/40-F accession number for a CIK."""
    # Use SEC submissions API
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=UA, timeout=30)
    if r.status_code != 200:
        return None
    
    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    
    # Find latest 20-F or 40-F
    for i, form in enumerate(forms):
        if form in ("20-F", "40-F", "20-F/A", "40-F/A"):
            # Return accession without dashes for URL formatting
            acc = accessions[i].replace("-", "")
            return acc
    
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse 20-F/40-F HTML filings")
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--cik", required=True, help="CIK (with or without leading zeros)")
    parser.add_argument("--accession", help="Accession number (without dashes); if omitted, fetch latest 20-F/40-F")
    parser.add_argument("--primary-doc", help="Primary document filename (if known)")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()
    
    # Normalize CIK
    cik = args.cik.zfill(10)
    
    # Get accession if not provided
    accession = args.accession
    if not accession:
        print(f"Fetching latest 20-F/40-F for CIK {cik}...")
        accession = get_latest_20f_accession(cik)
        if not accession:
            print("No 20-F/40-F found")
            return 1
        print(f"Found accession: {accession}")
    
    # Parse
    print(f"Parsing 20-F/40-F: CIK={cik}, Accession={accession}")
    results = parse_20f_filing(cik, accession, args.primary_doc)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return 1
    
    # Print results
    print("\nExtracted Financials (3 most recent periods):")
    for key, vals in results.items():
        if key in ("cik", "accession", "filing_type", "period_ends"):
            continue
        print(f"  {key}: {vals}")
    
    if results.get("period_ends"):
        print(f"\nPeriod ends: {results['period_ends']}")
    
    # Save if requested
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())