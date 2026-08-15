#!/usr/bin/env python3
"""
extract_investment_holdings.py — Extract investment holdings from SEC companyfacts API.

For each ticker, for each concept in a predefined list of investment-related XBRL concepts,
extract the historical USD values (if available) from the companyfacts endpoint.
Output a tidy DataFrame: [ticker, as_of_date, concept, value].

This captures aggregated totals like MarketableSecuritiesCurrent, MarketableSecuritiesNoncurrent,
AvailableForSaleSecurities, HeldToMaturitySecurities, etc., which are reliably reported in XBRL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
UA = {"User-Agent": "personal-research derek.moore@example.com"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# List of investment-related concepts we want to extract.
# These are drawn from us-gaap taxonomy and are commonly present in companyfacts.
INVESTMENT_CONCEPTS = {
    # Marketable securities (current and non-current)
    "MarketableSecuritiesCurrent",
    "MarketableSecuritiesNoncurrent",
    "MarketableSecurities",
    # Available for sale and held to maturity debt securities
    "AvailableForSaleSecurities",
    "AvailableForSaleDebtSecurities",
    "HeldToMaturitySecurities",
    "HeldToMaturityDebtSecurities",
    # Trading securities
    "TradingSecurities",
    # Short-term and long-term investments
    "ShortTermInvestments",
    "LongTermInvestments",
    "OtherInvestments",
    # Equity method and cost method investments
    "EquityMethodInvestments",
    "CostMethodInvestments",
    # Investments and advances
    "Investments",
    "InvestmentsAndAdvances",
    # Other assets (sometimes used for investments)
    "OtherAssets",
    "OtherCurrentAssets",
    "OtherNoncurrentAssets",
    # Investments in debt and equity securities (more detailed)
    "InvestmentsInDebtSecurities",
    "InvestmentsInEquitySecurities",
}

def load_cik_map() -> Dict[str, str]:
    r = requests.get(TICKERS_URL, headers=UA, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        cik = str(row["cik_str"]).zfill(10)
        out[str(row["ticker"]).upper()] = cik
    return out


def get_companyfacts(cik: str) -> Optional[Dict]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  Failed to fetch companyfacts for CIK {cik}: {e}")
        return None


def extract_investment_facts(facts: Dict) -> List[Dict]:
    """
    Extract investment-related facts from the companyfacts JSON.
    Returns a list of dicts: {concept, end_date, value, accn?}
    We focus on USD units.
    """
    results = []
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in INVESTMENT_CONCEPTS:
        if concept not in us_gaap:
            continue
        concept_data = us_gaap[concept]
        # Look for USD units
        units = concept_data.get("units", {})
        usd_entries = units.get("USD", [])
        if not usd_entries:
            # Sometimes the unit might be 'USD' or 'usd' but we already checked uppercase.
            # Try case-insensitive?
            for ukey, entries in units.items():
                if ukey.upper() == "USD":
                    usd_entries = entries
                    break
        if not usd_entries:
            continue
        # Sort by end date to get the most recent first (though we want all)
        # We'll keep all entries.
        for entry in usd_entries:
            # entry has: end, val, accn, fy, fp, form, filed
            # We want the end date as the as_of_date.
            end_date = entry.get("end")
            val = entry.get("val")
            if end_date is None or val is None:
                continue
            results.append({
                "concept": concept,
                "as_of_date": end_date,
                "value": float(val),
                # Optionally keep accession number for debugging
                # "accn": entry.get("accn"),
            })
    return results


def process_ticker(ticker: str, cik_map: Dict[str, str]) -> List[Dict]:
    cik = cik_map.get(ticker.upper())
    if not cik:
        print(f"[{ticker}] CIK not found")
        return []
    # print(f"[{ticker}] Processing CIK {cik}")
    facts = get_companyfacts(cik)
    if not facts:
        return []
    investment_facts = extract_investment_facts(facts)
    if not investment_facts:
        # print(f"[{ticker}] No investment facts found")
        return []
    # Enrich with ticker
    for f in investment_facts:
        f["ticker"] = ticker.upper()
    return investment_facts


def main():
    parser = argparse.ArgumentParser(description="Extract investment holdings from SEC companyfacts API")
    parser.add_argument("--tickers", nargs="+", help="List of tickers to process")
    parser.add_argument("--max-tickers", type=int, default=50, help="Maximum number of tickers to process (if --tickers not provided)")
    parser.add_argument("--output", default="investment_holdings.parquet", help="Output file for holdings data")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep between requests (seconds)")
    args = parser.parse_args()

    # Load universe of tickers if not provided
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        # Read from monitored_stocks.parquet or fundamentals.parquet
        if STOCKS_FILE.exists():
            stocks_df = pd.read_parquet(STOCKS_FILE)
            tickers = stocks_df['ticker'].unique().tolist()
        else:
            fundamentals_df = pd.read_parquet(FUND)
            tickers = fundamentals_df['ticker'].unique().tolist()
        if args.max_tickers:
            tickers = tickers[:args.max_tickers]

    print(f"Processing {len(tickers)} tickers: {tickers[:10]}{'...' if len(tickers) > 10 else ''}")

    # Load CIK map
    cik_map = load_cik_map()
    print(f"Loaded CIK map for {len(cik_map)} tickers")

    # Process each ticker
    all_results = []
    for i, ticker in enumerate(tickers, 1):
        if i % 20 == 0 or i == len(tickers):
            print(f"  Processed {i}/{len(tickers)} tickers")
        holdings = process_ticker(ticker, cik_map)
        all_results.extend(holdings)
        time.sleep(args.sleep)

    if not all_results:
        print("No investment holdings extracted")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    # Ensure as_of_date is datetime
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    # Sort by ticker, as_of_date, concept
    df = df.sort_values(["ticker", "as_of_date", "concept"]).reset_index(drop=True)
    print(f"Extracted {len(df)} investment holding facts")

    # Save to parquet
    output_path = DATA_DIR / args.output
    df.to_parquet(output_path)
    print(f"Saved to {output_path}")

    # Show summary
    print("\nSummary:")
    print(f"  Unique tickers: {df['ticker'].nunique()}")
    print(f"  Unique concepts: {df['concept'].nunique()}")
    print(f"  Date range: {df['as_of_date'].min()} to {df['as_of_date'].max()}")
    # Show top concepts by number of non-null entries
    top_concepts = df.groupby('concept').size().sort_values(ascending=False).head(10)
    print("\nTop 10 concepts by number of entries:")
    for concept, count in top_concepts.items():
        print(f"  {concept}: {count}")

if __name__ == "__main__":
    main()