#!/usr/bin/env python3
"""
Extract historical 13F-HR filings for all available quarters.

Builds quarterly ownership network panel: filer × as_of_date × held_ticker × shares/market_value
"""

import pandas as pd
import requests
import xml.etree.ElementTree as ET
import time
import json
from pathlib import Path
from tqdm import tqdm
from datetime import date

# Load CIK map
with open("cik_ticker_map.json") as f:
    CIK_MAP = {k.upper(): str(v).zfill(10) for k, v in json.load(f).items()}

CIK_TO_TICKER = {v: k for k, v in CIK_MAP.items()}

HEADERS = {'User-Agent': 'personal-research derek.moore@example.com'}
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

def get_13f_filings(cik: str, max_filings: int = None) -> list:
    """Get all 13F-HR filings for a CIK"""
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return []
        data = resp.json()
        recent = data['filings']['recent']
        forms = recent['form']
        accessions = recent['accessionNumber']
        report_dates = recent['reportDate']
        filing_dates = recent['filingDate']
        
        filings = []
        for i, (form, acc, rpt_dt, file_dt) in enumerate(zip(forms, accessions, report_dates, filing_dates)):
            if '13F-HR' in form and not form.endswith('/A'):  # Skip amendments
                filings.append({
                    'accession': acc,
                    'report_date': rpt_dt,
                    'filing_date': file_dt
                })
        
        # Sort by report date descending
        filings.sort(key=lambda x: x['report_date'], reverse=True)
        
        if max_filings:
            filings = filings[:max_filings]
        
        return filings
    except Exception as e:
        print(f"  Error getting filings for {cik}: {e}")
        return []

def fetch_13f_information_table(cik: str, accession: str) -> str:
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
    
    # Fallback patterns
    if not info_table_url:
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
    
    # Last resort: filing directory listing
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

def parse_13f_xml(xml_content: str) -> list:
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

def extract_filer_history(ticker: str, max_filings: int = None) -> list:
    """Extract all historical 13F-HR for one filer"""
    cik = CIK_MAP[ticker]
    filings = get_13f_filings(cik, max_filings)
    
    all_holdings = []
    for filing in tqdm(filings, desc=f"  {ticker}", leave=False):
        xml_content = fetch_13f_information_table(cik, filing['accession'])
        if xml_content:
            holdings = parse_13f_xml(xml_content)
            for h in holdings:
                h['filer_ticker'] = ticker
                h['filer_cik'] = cik
                h['as_of_date'] = filing['report_date']
                h['filing_date'] = filing['filing_date']
                h['accession'] = filing['accession']
                h['form'] = '13F-HR'
                h['source_form_type'] = '13F-HR'
            all_holdings.extend(holdings)
            print(f"    {filing['report_date']}: {len(holdings)} holdings")
        time.sleep(0.1)  # Rate limit
    
    return all_holdings

def main():
    # Target filers with good 13F-HR history
    target_filers = [
        'BRK-B', 'JPM', 'GS', 'BAC', 'C', 'WFC', 'MS', 'BLK', 'V', 'AXP', 
        'COF', 'USB', 'PNC', 'TFC', 'CB', 'MET', 'PRU', 'AIG', 'ALL', 'TRV',
        'CINF', 'WRB', 'AFG', 'EQH', 'LNC', 'CNO', 'FNF', 'FAF', 'RNR',
        'INTC', 'AMD', 'CSCO', 'CRM', 'PLTR', 'UBER', 'GOOGL', 'AMZN', 'NVDA'
    ]
    
    print(f"Extracting historical 13F-HR for {len(target_filers)} filers...")
    
    all_holdings = []
    for ticker in tqdm(target_filers, desc="Filers"):
        try:
            holdings = extract_filer_history(ticker, max_filings=None)  # All available
            all_holdings.extend(holdings)
            print(f"  {ticker}: {len(holdings)} total holdings across all quarters")
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")
        time.sleep(0.5)  # Rate limit between filers
    
    # Convert to DataFrame
    if all_holdings:
        df = pd.DataFrame(all_holdings)
        
        # Clean up
        df['as_of_date'] = pd.to_datetime(df['as_of_date']).dt.date
        df['filing_date'] = pd.to_datetime(df['filing_date']).dt.date
        
        # Numeric fields
        for col in ['sshPrnamt', 'value']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rename for consistency
        df = df.rename(columns={
            'nameOfIssuer': 'held_nameOfIssuer',
            'titleOfClass': 'held_titleOfClass',
            'cusip': 'held_cusip',
            'sshPrnamt': 'held_shares',
            'value': 'held_value_thousands'  # 13F values are in $1000s
        })
        
        # Save
        df.to_parquet('historical_13f_holdings.parquet', index=False)
        print(f"\nSaved {len(df)} historical holdings to historical_13f_holdings.parquet")
        print(f"Date range: {df['as_of_date'].min()} to {df['as_of_date'].max()}")
        print(f"Unique quarters: {df['as_of_date'].nunique()}")
        print(f"Filers: {df['filer_ticker'].nunique()}")
        
        # Summary by quarter
        print("\nHoldings per quarter:")
        print(df.groupby('as_of_date').size().sort_index())
        
        # Summary by filer
        print("\nHoldings per filer:")
        print(df.groupby('filer_ticker').size().sort_values(ascending=False))
    else:
        print("No holdings extracted!")

if __name__ == "__main__":
    main()