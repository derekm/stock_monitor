#!/usr/bin/env python3
"""
backfill_holdings.py — Pull 13F-HR holdings from SEC EDGAR to build ownership network.

For each institutional manager (filers with 13F-HR filings), extracts:
  filer_ticker, period_end, held_cusip, shares, market_value (as of filing date).

Then maps held_cusip to ticker using a crosswalk (built from SEC company_tickers + CUSIP mapping?).
If mapping fails, holds by CUSIP.

Outputs:
  holdings.parquet — filer × as_of_date × held_ticker/cusip × shares, market_value
  holdings_daily_value.parquet — filer × as_of_date × held_ticker × daily_market_value (shares * price)
  filer_holdings_value.parquet — filer × as_of_date × total_holdings_market_value

Usage:
  python backfill_holdings.py --max-filers 10
  python backfill_holdings.py --filers BRK,AAPL,MSFT
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
import xml.etree.ElementTree as ET

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
OUT_HOLDINGS = DATA_DIR / "holdings.parquet"
OUT_HOLDINGS_DAILY = DATA_DIR / "holdings_daily_value.parquet"
OUT_FILER_VALUE = DATA_DIR / "filer_holdings_value.parquet"

# SEC headers
UA = {"User-Agent": "personal-research derek.moore@example.com"}
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
# Base for archived filings
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"

# We'll need a CUSIP to ticker mapping. Build from SEC's company_tickers + maybe crosswalk?
# For simplicity, we'll attempt to map using the ticker from the held security's name? Not reliable.
# Instead, we'll output holdings by CUSIP and later join to fundamentals via a separate CUSIP crosswalk
# that we can build from the SEC's daily list? Not available.
# Alternative: use the fact that many 13F filings also include the ticker in the <issueName> or <titleOfClass>?
# The 13F XML includes <titleOfClass> and sometimes the ticker is in there.
# We'll extract both CUSIP and ticker if available.

def get_cik_map() -> Dict[str, str]:
    """Map ticker -> CIK (10-digit zero-padded) from SEC."""
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().values():
        ticker = str(row["ticker"]).upper()
        cik = str(row["cik_str"]).zfill(10)
        out[ticker] = cik
    return out

def get_submissions(cik: str) -> dict:
    url = SUBMISSIONS_URL.format(cik=cik)
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()

def extract_13f_holdings(cik: str, accession_no: str, primary_doc: str) -> List[dict]:
    """
    Given a 13F-HR filing, download the XML and parse holdings.
    accession_no: includes dashes (e.g., 0001104659-23-012345)
    primary_doc: filename.xml
    """
    # Format accession number for URL: remove dashes
    acc_no_nodash = accession_no.replace("-", "")
    url = f"{ARCHIVE_BASE}/{int(cik)}/{acc_no_nodash}/{primary_doc}"
    # Try with .xml extension if not already
    if not primary_doc.endswith(".xml"):
        url = f"{ARCHIVE_BASE}/{int(cik)}/{acc_no_nodash}/{primary_doc}.xml"
    r = requests.get(url, headers=UA, timeout=30)
    if r.status_code != 200:
        # Try without .xml if we added it
        if primary_doc.endswith(".xml"):
            url2 = f"{ARCHIVE_BASE}/{int(cik)}/{acc_no_nodash}/{primary_doc[:-4]}"
            r = requests.get(url2, headers=UA, timeout=30)
            if r.status_code != 200:
                print(f"  !! Failed to download 13F XML: {url}")
                return []
        else:
            print(f"  !! Failed to download 13F XML: {url}")
            return []
    # Parse XML
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError as e:
        print(f"  !! XML parse error: {e}")
        return []
    # Define namespace
    ns = {"": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}
    holdings = []
    for info_table in root.findall(".//infoTable", ns):
        # Extract fields
        cusip_el = info_table.find("cusip", ns)
        ticker_el = info_table.find("ticker", ns)  # some filers include ticker
        shares_el = info_table.find("shrsOrPrnAmt", ns)
        if shares_el is not None:
            shrs_el = shares_el.find("sshPrnamt", ns)
            type_el = shares_el.find("sshPrnamtType", ns)
        else:
            shrs_el = None
            type_el = None
        market_value_el = info_table.find("value", ns)  # in thousands
        put_call_el = info_table.find("putCall", ns)
        # We only want equity holdings (not put/call)
        if put_call_el is not None and put_call_el.text and put_call_el.text.upper() in ("PUT", "CALL"):
            continue
        cusip = cusip_el.text if cusip_el is not None else None
        ticker = ticker_el.text if ticker_el is not None else None
        shares = float(shrs_el.text) if shrs_el is not None and shrs_el.text else None
        shares_type = type_el.text if type_el is not None else None
        market_value = float(market_value_el.text) * 1000 if market_value_el is not None and market_value_el.text else None  # value is in thousands
        # Only keep if we have shares and cusip
        if cusip and shares is not None:
            holdings.append({
                "held_cusip": cusip,
                "held_ticker": ticker,
                "shares": shares,
                "shares_type": shares_type,
                "market_value_reported": market_value,
            })
    return holdings

def build_holdings_for_filer(filer_ticker: str, cik: str, max_filings: int = 5) -> List[dict]:
    """Fetch recent 13F-HR filings for a CIK and extract holdings."""
    try:
        subs = get_submissions(cik)
    except Exception as e:
        print(f"  !! Failed to get submissions for {filer_ticker} (CIK {cik}): {e}")
        return []
    # Get recent filings
    recent = subs.get("filings", {}).get("recent", {})
    if not recent:
        print(f"  !! No recent filings for {filer_ticker}")
        return []
    form = recent.get("form", [])
    filing_dates = recent.get("filed", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    # Filter for 13F-HR
    holdings_all = []
    for i, f in enumerate(form):
        if f.strip().upper() == "13F-HR":
            date_str = filing_dates[i]
            acc = accession_numbers[i]
            prim = primary_docs[i]
            try:
                holdings = extract_13f_holdings(cik, acc, prim)
                for h in holdings:
                    h["filer_ticker"] = filer_ticker
                    h["period_end"] = date_str  # YYYY-MM-DD
                holdings_all.extend(holdings)
                print(f"    {filer_ticker}: {len(holdings)} holdings from {date_str}")
                if len(holdings_all) >= max_filings * 50:  # rough limit
                    break
            except Exception as e:
                print(f"    !! Error processing 13F for {filer_ticker} {date_str}: {e}")
    return holdings_all

def main():
    ap = argparse.ArgumentParser(description="Build ownership network from 13F-HR filings")
    ap.add_argument("--max-filers", type=int, default=0, help="Maximum number of filers to process (0 = all)")
    ap.add_argument("--filers", type=str, default=None, help="Comma-separated list of ticker filers to process")
    ap.add_argument("--max-filings-per-filer", type=int, default=3, help="Maximum 13F-HR filings per filer to process")
    args = ap.parse_args()

    # Get universe of tickers (from fundamentals or monitored_stocks)
    if FUND.exists():
        fund = pd.read_parquet(FUND)
        tickers = sorted(fund["ticker"].unique())
    else:
        stocks = pd.read_parquet(DATA_DIR / "monitored_stocks.parquet")
        tickers = sorted(stocks["ticker"].unique())
    print(f"Universe: {len(tickers)} tickers")

    if args.filers:
        filers = [t.strip().upper() for t in args.filers.split(",") if t.strip()]
    else:
        filers = tickers
    if args.max_filers > 0:
        filers = filers[:args.max_filers]

    # Get CIK map
    print("Loading ticker->CIK map...")
    cik_map = get_cik_map()
    print(f"Mapped {len(cik_map)} tickers to CIK")

    all_holdings = []
    processed = 0
    for filer in filers:
        cik = cik_map.get(filer)
        if not cik:
            print(f"  {filer}: no CIK found, skipping")
            continue
        print(f"Processing {filer} (CIK {cik})...")
        holdings = build_holdings_for_filer(filer, cik, max_filings=args.max_filings_per_filer)
        if holdings:
            all_holdings.extend(holdings)
            processed += 1
        time.sleep(0.2)  # be nice to SEC

    if not all_holdings:
        print("No holdings data fetched.")
        return

    # Convert to DataFrame
    holdings_df = pd.DataFrame(all_holdings)
    print(f"\nTotal holdings records: {len(holdings_df)}")
    print(f"Unique filers: {holdings_df['filer_ticker'].nunique()}")
    print(f"Unique held CUSIPs: {holdings_df['held_cusip'].nunique()}")
    if holdings_df['held_ticker'].notna().any():
        print(f"Unique held tickers (when available): {holdings_df['held_ticker'].notna().sum()}")

    # Save raw holdings
    holdings_df.to_parquet(OUT_HOLDINGS, index=False)
    print(f"Saved holdings → {OUT_HOLDINGS}")

    # TODO: Map held_cusip to ticker using crosswalk (future work)
    # For now, we'll compute daily market value only for those holdings where we have ticker
    # We'll need to join with daily_prices/ on ticker and date.

    # Load daily prices
    if not (DATA_DIR / "daily_prices/").exists():
        print("daily_prices/ not found; skipping daily value calculation")
        return
    prices = pd.read_parquet(DATA_DIR / "daily_prices/")
    prices = prices.rename(columns={"date": "as_of_date", "close": "price"})
    prices["as_of_date"] = pd.to_datetime(prices["as_of_date"])
    # We need to map held_cusip to ticker. Since we don't have a map, we'll skip for now.
    # Instead, we'll output a warning and compute only for holdings with ticker.
    if holdings_df['held_ticker'].notna().any():
        # Merge on ticker and date
        # Note: holdings_df period_end is string, need to convert
        holdings_df["period_end"] = pd.to_datetime(holdings_df["period_end"])
        # For each holding, we need the price of the held ticker on the period_end date (or last price <= period_end)
        # We'll do a merge_asof per ticker? Simpler: we'll compute the market value as shares * price where price is the last price <= period_end.
        # We'll do a simple merge and then fill forward/backward per ticker.
        # For simplicity, we'll just take the price on the exact period_end date if available.
        # This is approximate but okay for demonstration.
        merged = holdings_df.merge(
            prices,
            left_on=["held_ticker", "period_end"],
            right_on=["ticker", "as_of_date"],
            how="left",
            suffixes=("", "_held")
        )
        # If price missing, try to get previous close? We'll just drop missing for now.
        merged["daily_market_value"] = merged["shares"] * merged["price"]
        # Keep only rows where we have price
        merged = merged[merged["price"].notna()]
        print(f"Holdings with matched price: {len(merged)} / {len(holdings_df)}")
        if len(merged) > 0:
            # Save holdings daily value
            merged[["filer_ticker", "period_end", "held_ticker", "shares", "price", "daily_market_value"]].to_parquet(OUT_HOLDINGS_DAILY, index=False)
            print(f"Saved holdings daily value → {OUT_HOLDINGS_DAILY}")
            # Aggregate to filer level
            filer_value = merged.groupby(["filer_ticker", "period_end"])["daily_market_value"].sum().reset_index()
            filer_value.columns = ["filer_ticker", "as_of_date", "total_holdings_market_value"]
            filer_value.to_parquet(OUT_FILER_VALUE, index=False)
            print(f"Saved filer holdings value → {OUT_FILER_VALUE}")
            # Show sample
            print("\nSample filer holdings value:")
            print(filer_value.head(10))
        else:
            print("No holdings could be matched to daily prices (ticker missing or date mismatch).")
    else:
        print("No held ticker available in 13F data (need to map CUSIP to ticker).")

if __name__ == "__main__":
    main()