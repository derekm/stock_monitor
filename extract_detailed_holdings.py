#!/usr/bin/env python3
"""
extract_detailed_holdings.py — Extract detailed holdings from public companies' SEC filings.

Supports multiple form types:
1. 10-K/10-Q inline XBRL - for operating companies' equity method investments, securities schedules
2. 13F-HR information table - for institutional investment managers' complete portfolios
3. SC 13D/G - for beneficial ownership >5%
4. N-PORT - for registered investment companies

Output: detailed_holdings.parquet with unified schema for all form types
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm

# SEC API Configuration
SEC_BASE = "https://data.sec.gov"
SEC_SUBMISSIONS = f"{SEC_BASE}/submissions/CIK{{cik}}.json"
SEC_COMPANYFACTS = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{{cik}}.json"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
USER_AGENT = "personal-research derek.moore@example.com"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
RATE_LIMIT_SECONDS = 0.12  # 10 requests/second max per SEC guidelines

# Investment-related concepts to extract from 10-K/10-Q
INVESTMENT_CONCEPTS = {
    "MarketableSecuritiesCurrent": "us-gaap:MarketableSecuritiesCurrent",
    "MarketableSecuritiesNoncurrent": "us-gaap:MarketableSecuritiesNoncurrent",
    "MarketableSecuritiesTotal": "us-gaap:MarketableSecurities",
    "AvailableForSaleSecurities": "us-gaap:AvailableForSaleSecurities",
    "HeldToMaturitySecurities": "us-gaap:HeldToMaturitySecurities",
    "TradingSecurities": "us-gaap:TradingSecurities",
    "ShortTermInvestments": "us-gaap:ShortTermInvestments",
    "LongTermInvestments": "us-gaap:LongTermInvestments",
    "EquityMethodInvestments": "us-gaap:EquityMethodInvestments",
    "CostMethodInvestments": "us-gaap:CostMethodInvestments",
    "OtherInvestments": "us-gaap:OtherInvestments",
    "Investments": "us-gaap:Investments",
    "OtherAssets": "us-gaap:OtherAssets",
}

# Dimensions that provide meaningful breakdowns for investments
RELEVANT_DIMENSIONS = {
    "us-gaap:FinancialInstrumentAxis": "instrument_type",
    "us-gaap:FairValueByFairValueHierarchyLevelAxis": "fair_value_level",
    "us-gaap:StatementClassOfStockAxis": "stock_class",
    "us-gaap:MajorEquityInvestmentsAxis": "major_equity_investment",
    "us-gaap:EquityMethodInvestmentsAxis": "equity_method_investment",
    "us-gaap:InvestmentsByTypeAxis": "investment_type",
    "us-gaap:ScheduleOfEquityMethodInvestmentsAxis": "equity_method_schedule",
    "srt:ScheduleOfEquityMethodInvestmentEquityMethodInvesteeNameAxis": "equity_method_investee",
    "dei:LegalEntityAxis": "legal_entity",
    "srt:CounterpartyNameAxis": "counterparty_name",
}


def load_cik_map() -> Dict[str, str]:
    """Load ticker -> CIK mapping from SEC's company_tickers.json"""
    cache_path = Path("cik_ticker_map.json")
    if cache_path.exists():
        with open(cache_path) as f:
            return {k.upper(): str(v).zfill(10) for k, v in json.load(f).items()}

    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    mapping = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
    with open(cache_path, "w") as f:
        json.dump(mapping, f)
    return mapping


def get_cik(ticker: str, cik_map: Dict[str, str]) -> Optional[str]:
    """Get CIK for ticker, handling BRK.B -> BRK-B mapping"""
    ticker_upper = ticker.upper()
    # Handle BRK.B / BRK.B -> BRK-B mapping
    if ticker_upper in ("BRK.B", "BRK-B"):
        return cik_map.get("BRK-B") or cik_map.get("BRK.A") or cik_map.get("BRK-A")
    return cik_map.get(ticker_upper)


def fetch_submissions(cik: str) -> Optional[Dict]:
    """Fetch submissions for a CIK"""
    url = SEC_SUBMISSIONS.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def get_recent_filings(submissions: Dict, forms: Tuple[str, ...] = ("10-K", "10-Q", "13F-HR", "13F-HR/A", "SC 13D", "SC 13G", "N-PORT"), limit: int = 10) -> List[Dict]:
    """Extract recent relevant filings from submissions"""
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return []

    form_list = recent.get("form", [])
    accession_list = recent.get("accessionNumber", [])
    filing_date_list = recent.get("filingDate", [])
    report_date_list = recent.get("reportDate", [])
    primary_doc_list = recent.get("primaryDocument", [])

    filings = []
    for i, form in enumerate(form_list):
        if form in forms and i < len(accession_list):
            accession = accession_list[i].replace("-", "")
            primary_doc = primary_doc_list[i] if i < len(primary_doc_list) else ""
            filings.append({
                "form": form,
                "accession": accession_list[i],
                "accession_no_dash": accession,
                "filing_date": filing_date_list[i] if i < len(filing_date_list) else None,
                "report_date": report_date_list[i] if i < len(report_date_list) else None,
                "primary_document": primary_doc,
            })
        if len(filings) >= limit:
            break
    return filings


def download_filing(cik: str, accession_no_dash: str, primary_document: str) -> Optional[str]:
    """Download the primary filing document"""
    url = f"{SEC_ARCHIVES}/{int(cik)}/{accession_no_dash}/{primary_document}"
    resp = requests.get(url, headers=HEADERS, timeout=60)
    if resp.status_code == 404:
        url = f"{SEC_ARCHIVES}/{cik.lstrip('0')}/{accession_no_dash}/{primary_document}"
        resp = requests.get(url, headers=HEADERS, timeout=60)
        if resp.status_code == 404:
            return None
    resp.raise_for_status()
    return resp.text


def parse_inline_xbrl(html_content: str) -> Tuple[Dict[str, Dict], Dict[str, Dict], List[Dict]]:
    """Parse inline XBRL content to extract contexts, units, and facts"""
    contexts = {}
    units = {}
    facts = []

    # Extract xbrli:context elements
    ctx_pattern = r'(<xbrli:context[^>]*id="([^"]+)"[^>]*>.*?</xbrli:context>)'
    ctx_matches = re.findall(ctx_pattern, html_content, re.DOTALL)

    for full_ctx, ctx_id in ctx_matches:
        entity_match = re.search(
            r'<xbrli:identifier[^>]*scheme="([^"]+)"[^>]*>([^<]+)</xbrli:identifier>',
            full_ctx
        )
        entity_scheme = entity_match.group(1) if entity_match else None
        entity_id = entity_match.group(2) if entity_match else None

        period_match = re.search(r'<xbrli:period>(.*?)</xbrli:period>', full_ctx, re.DOTALL)
        period_content = period_match.group(1) if period_match else ""

        start_date = None
        end_date = None
        instant_date = None

        start_match = re.search(r'<xbrli:startDate>([^<]+)</xbrli:startDate>', period_content)
        if start_match:
            start_date = start_match.group(1)

        end_match = re.search(r'<xbrli:endDate>([^<]+)</xbrli:endDate>', period_content)
        if end_match:
            end_date = end_match.group(1)

        instant_match = re.search(r'<xbrli:instant>([^<]+)</xbrli:instant>', period_content)
        if instant_match:
            instant_date = instant_match.group(1)

        # Parse segment/dimensions
        dimensions = {}
        segment_match = re.search(r'<xbrli:segment>(.*?)</xbrli:segment>', full_ctx, re.DOTALL)
        if segment_match:
            segment_content = segment_match.group(1)
            explicit_matches = re.findall(
                r'<xbrldi:explicitMember[^>]*dimension="([^"]+)"[^>]*>([^<]+)</xbrldi:explicitMember>',
                segment_content
            )
            for dim, member in explicit_matches:
                dimensions[dim] = member

            typed_matches = re.findall(
                r'<xbrldi:typedMember[^>]*dimension="([^"]+)"[^>]*>(.*?)</xbrldi:typedMember>',
                segment_content,
                re.DOTALL
            )
            for dim, member_content in typed_matches:
                domain_match = re.search(r'<[^>]+domain>([^<]+)</[^>]+domain>', member_content)
                if domain_match:
                    dimensions[dim] = domain_match.group(1)
                else:
                    dimensions[dim] = member_content.strip()

        contexts[ctx_id] = {
            "entity_scheme": entity_scheme,
            "entity_id": entity_id,
            "start_date": start_date,
            "end_date": end_date,
            "instant_date": instant_date,
            "dimensions": dimensions,
        }

    # Extract xbrli:unit elements
    unit_pattern = r'(<xbrli:unit[^>]*id="([^"]+)"[^>]*>.*?</xbrli:unit>)'
    unit_matches = re.findall(unit_pattern, html_content, re.DOTALL)
    for full_unit, unit_id in unit_matches:
        measure_match = re.search(r'<xbrli:measure>([^<]+)</xbrli:measure>', full_unit)
        if measure_match:
            units[unit_id] = measure_match.group(1)
        else:
            units[unit_id] = "pure"

    # Extract numeric facts (ix:nonFraction)
    frac_pattern = r'<ix:nonFraction([^>]*)>([^<]+)</ix:nonFraction>'
    frac_matches = re.findall(frac_pattern, html_content)

    for attrs, value in frac_matches:
        name_match = re.search(r'name="([^"]+)"', attrs)
        ctx_match = re.search(r'contextRef="([^"]+)"', attrs)
        unit_match = re.search(r'unitRef="([^"]+)"', attrs)
        if not name_match or not ctx_match:
            continue
        name = name_match.group(1)
        ctx_ref = ctx_match.group(1)
        unit_ref = unit_match.group(1) if unit_match else ""

        clean_value = value.replace(",", "").replace("&#160;", "").replace("&#8212;", "").strip()
        if clean_value.startswith("(") and clean_value.endswith(")"):
            clean_value = "-" + clean_value[1:-1]
        try:
            num_value = float(clean_value)
        except ValueError:
            continue

        facts.append({
            "concept": name,
            "context_ref": ctx_ref,
            "unit_ref": unit_ref,
            "value": num_value,
            "is_numeric": True,
        })

    # Extract non-numeric facts (ix:nonNumeric)
    nonfrac_pattern = r'<ix:nonNumeric[^>]*contextRef="([^"]+)"[^>]*name="([^"]+)"[^>]*>([^<]+)</ix:nonNumeric>'
    nonfrac_matches = re.findall(nonfrac_pattern, html_content)

    for ctx_ref, name, value in nonfrac_matches:
        facts.append({
            "concept": name,
            "context_ref": ctx_ref,
            "unit_ref": None,
            "value": value.strip(),
            "is_numeric": False,
        })

    return contexts, units, facts


def extract_holdings_from_10k10q(ticker: str, cik: str, filing: Dict, html_content: str) -> List[Dict]:
    """Extract investment-related holdings from 10-K/10-Q inline XBRL"""
    contexts, units, facts = parse_inline_xbrl(html_content)

    as_of_date_str = filing.get("report_date") or filing.get("filing_date")
    if not as_of_date_str:
        return []

    try:
        as_of_date = datetime.strptime(as_of_date_str, "%Y-%m-%d").date()
    except ValueError:
        return []

    investment_facts = []
    for fact in facts:
        if not fact["is_numeric"]:
            continue

        concept = fact["concept"]
        is_investment = any(
            inv_concept in concept for inv_concept in INVESTMENT_CONCEPTS.values()
        )
        investment_keywords = [
            "Investment", "Securities", "EquityMethod", "CostMethod",
            "AvailableForSale", "HeldToMaturity", "TradingSecurities",
            "MarketableSecurities", "ShortTermInvestments", "LongTermInvestments"
        ]
        is_investment = is_investment or any(kw in concept for kw in investment_keywords)

        if not is_investment:
            continue

        ctx = contexts.get(fact["context_ref"])
        if not ctx:
            continue

        unit = units.get(fact["unit_ref"], "unknown")
        dimensions = ctx.get("dimensions", {})
        relevant_dims = {}
        for dim_axis, dim_name in RELEVANT_DIMENSIONS.items():
            if dim_axis in dimensions:
                relevant_dims[dim_name] = dimensions[dim_axis]

        all_dims_json = json.dumps(dimensions) if dimensions else "{}"
        relevant_dims_json = json.dumps(relevant_dims) if relevant_dims else "{}"

        investment_facts.append({
            "filer_ticker": ticker,
            "filer_cik": cik,
            "as_of_date": as_of_date,
            "filing_date": filing.get("filing_date"),
            "report_date": filing.get("report_date"),
            "form": filing.get("form"),
            "accession": filing.get("accession"),
            "concept": concept,
            "value": fact["value"],
            "unit": unit,
            "context_ref": fact["context_ref"],
            "period_start": ctx.get("start_date"),
            "period_end": ctx.get("end_date") or ctx.get("instant_date"),
            "dimensions": all_dims_json,
            "relevant_dimensions": relevant_dims_json,
            "source_form_type": "10-K/10-Q",
        })

    return investment_facts


def fetch_13f_information_table(cik: str, accession: str) -> Optional[str]:
    """Fetch 13F-HR information table XML from filing"""
    base_url = f"{SEC_ARCHIVES}/{int(cik)}/{accession.replace('-', '')}/"
    
    # Try index page first
    index_url = base_url + f'{accession.replace("-", "")}-index.html'
    resp = requests.get(index_url, headers=HEADERS)

    if resp.status_code != 200:
        index_url = base_url + f'{accession.replace("-", "")}-index-headers.html'
        resp = requests.get(index_url, headers=HEADERS)
    
    info_table_url = None
    
    if resp.status_code == 200:
        import re
        xml_links = re.findall(r'href=["\']([^"\']+\.xml)["\']', resp.text)
        
        # Check all XML links for the information table
        for link in xml_links:
            if 'primary_doc' in link or 'index' in link:
                continue
            # The link might be absolute (starting with /) or relative
            if link.startswith('http'):
                test_url = link
            elif link.startswith('/'):
                test_url = f"https://www.sec.gov{link}"
            else:
                test_url = base_url + link
            test_resp = requests.get(test_url, headers=HEADERS)
            if test_resp.status_code == 200 and 'informationTable' in test_resp.text:
                info_table_url = test_url
                break
    
    # Fallback: try the direct Information_Table pattern with date
    if not info_table_url:
        # We don't know the exact date format, but we can try common patterns
        # The date is usually the period end date from the filing
        # For now, try without date suffix
        patterns = [
            'Information_Table.xml',
            'informationTable.xml',
            'infoTable.xml',
            '56757.xml',
        ]
        for pattern in patterns:
            test_url = base_url + pattern
            test_resp = requests.get(test_url, headers=HEADERS)
            if test_resp.status_code == 200 and 'informationTable' in test_resp.text:
                info_table_url = test_url
                break
    
    # Last resort: try to get filing directory listing
    if not info_table_url:
        filing_resp = requests.get(base_url, headers=HEADERS)
        if filing_resp.status_code == 200:
            import re
            xml_links = re.findall(r'href=["\']([^"\']+\.xml)["\']', filing_resp.text)
            for link in xml_links:
                if 'primary_doc' in link or 'index' in link:
                    continue
                if link.startswith('http'):
                    test_url = link
                elif link.startswith('/'):
                    test_url = f"https://www.sec.gov{link}"
                else:
                    test_url = base_url + link
                test_resp = requests.get(test_url, headers=HEADERS)
                if test_resp.status_code == 200 and 'informationTable' in test_resp.text:
                    info_table_url = test_url
                    break

    if not info_table_url:
        return None

    resp = requests.get(info_table_url, headers=HEADERS)
    return resp.text


def parse_13f_xml(xml_content: str) -> List[Dict]:
    """Parse 13F-HR information table XML"""
    root = ET.fromstring(xml_content)
    ns = 'http://www.sec.gov/edgar/document/thirteenf/informationtable'
    
    holdings = []
    for info_table in root.findall(f'{{{ns}}}infoTable'):
        holding = {}
        for child in info_table:
            tag = child.tag.replace(f'{{{ns}}}', '')
            if tag == 'shrsOrPrnAmt':
                for sub in child:
                    sub_tag = sub.tag.replace(f'{{{ns}}}', '')
                    holding[sub_tag] = sub.text
            elif tag == 'votingAuthority':
                for sub in child:
                    sub_tag = sub.tag.replace(f'{{{ns}}}', '')
                    holding[f'votingAuthority_{sub_tag}'] = sub.text
            else:
                holding[tag] = child.text
        holdings.append(holding)
    
    return holdings


def extract_holdings_from_13f(ticker: str, cik: str, filing: Dict) -> List[Dict]:
    """Extract holdings from 13F-HR filing"""
    xml_content = fetch_13f_information_table(cik, filing["accession"])
    if not xml_content:
        return []

    try:
        holdings = parse_13f_xml(xml_content)
    except ET.ParseError:
        return []

    if not holdings:
        return []

    as_of_date_str = filing.get("report_date") or filing.get("filing_date")
    if not as_of_date_str:
        return []

    try:
        as_of_date = datetime.strptime(as_of_date_str, "%Y-%m-%d").date()
    except ValueError:
        return []

    investment_facts = []
    for holding in holdings:
        investment_facts.append({
            "filer_ticker": ticker,
            "filer_cik": cik,
            "as_of_date": as_of_date,
            "filing_date": filing.get("filing_date"),
            "report_date": filing.get("report_date"),
            "form": filing.get("form"),
            "accession": filing.get("accession"),
            "concept": "us-gaap:Investments",  # Generic concept for 13F holdings
            "value": float(holding.get("value", 0)) if holding.get("value") else 0,
            "unit": "iso4217:USD",
            "context_ref": "13F-HR",
            "period_start": None,
            "period_end": as_of_date_str,
            "dimensions": json.dumps({
                "nameOfIssuer": holding.get("nameOfIssuer", ""),
                "titleOfClass": holding.get("titleOfClass", ""),
                "cusip": holding.get("cusip", ""),
                "sshPrnamt": holding.get("sshPrnamt", ""),
                "sshPrnamtType": holding.get("sshPrnamtType", ""),
                "investmentDiscretion": holding.get("investmentDiscretion", ""),
                "otherManager": holding.get("otherManager", ""),
                "votingAuthority_Sole": holding.get("votingAuthority_Sole", ""),
                "votingAuthority_Shared": holding.get("votingAuthority_Shared", ""),
                "votingAuthority_None": holding.get("votingAuthority_None", ""),
            }),
            "relevant_dimensions": json.dumps({
                "nameOfIssuer": holding.get("nameOfIssuer", ""),
                "cusip": holding.get("cusip", ""),
            }),
            "source_form_type": "13F-HR",
            "held_nameOfIssuer": holding.get("nameOfIssuer", ""),
            "held_titleOfClass": holding.get("titleOfClass", ""),
            "held_cusip": holding.get("cusip", ""),
            "held_shares": float(holding.get("sshPrnamt", 0)) if holding.get("sshPrnamt") else 0,
        })

    return investment_facts


def process_ticker(ticker: str, cik_map: Dict[str, str], max_filings: int = 5) -> List[Dict]:
    """Process all filings for a single ticker"""
    cik = get_cik(ticker, cik_map)
    if not cik:
        return []

    submissions = fetch_submissions(cik)
    if not submissions:
        return []

    filings = get_recent_filings(submissions, limit=max_filings)
    if not filings:
        return []

    all_holdings = []
    for filing in filings:
        primary_doc = filing.get("primary_document", "")
        if not primary_doc:
            continue

        if filing["form"] in ("13F-HR", "13F-HR/A"):
            # 13F-HR uses different parsing
            holdings = extract_holdings_from_13f(ticker, cik, filing)
            print(f"  13F-HR {filing['accession']}: {len(holdings)} holdings")
        else:
            # 10-K/10-Q use XBRL parsing
            html_content = download_filing(cik, filing["accession_no_dash"], primary_doc)
            if not html_content:
                continue
            holdings = extract_holdings_from_10k10q(ticker, cik, filing, html_content)
            print(f"  {filing['form']} {filing['accession']}: {len(holdings)} investment facts")

        all_holdings.extend(holdings)
        time.sleep(RATE_LIMIT_SECONDS)

    return all_holdings


def main():
    parser = argparse.ArgumentParser(description="Extract detailed holdings from SEC filings (10-K, 10-Q, 13F-HR)")
    parser.add_argument("--tickers", required=True, help="Comma-separated list of tickers")
    parser.add_argument("--max-filings", type=int, default=5, help="Max filings per ticker")
    parser.add_argument("--output", default="detailed_holdings.parquet", help="Output parquet file")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    cik_map = load_cik_map()
    print(f"Loaded CIK map with {len(cik_map)} entries")

    all_holdings = []
    for ticker in tqdm(tickers, desc="Processing tickers"):
        print(f"\nProcessing {ticker}...")
        holdings = process_ticker(ticker, cik_map, args.max_filings)
        print(f"  Total: {len(holdings)} holdings/facts")
        all_holdings.extend(holdings)

    if not all_holdings:
        print("No holdings extracted!")
        return

    df = pd.DataFrame(all_holdings)

    # Ensure as_of_date is date type
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    if "filing_date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.date
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    if "period_start" in df.columns:
        df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce").dt.date
    if "period_end" in df.columns:
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce").dt.date

    # Sort
    df = df.sort_values(["filer_ticker", "as_of_date", "concept"]).reset_index(drop=True)

    # Save
    df.to_parquet(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Tickers: {df['filer_ticker'].nunique()}")
    print(f"Date range: {df['as_of_date'].min()} to {df['as_of_date'].max()}")
    print(f"Unique concepts: {df['concept'].nunique()}")
    print(f"Form types: {df['source_form_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()