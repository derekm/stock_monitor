#!/usr/bin/env python3
"""
parse_13f_hr.py — Parse SEC Form 13F-HR information table XML.

Extracts issuer-level holdings from 13F-HR filings for institutional investment managers.
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import time
import json

HEADERS = {'User-Agent': 'personal-research derek.moore@example.com'}

def fetch_13f_holdings(cik: str, accession: str) -> pd.DataFrame:
    """
    Fetch and parse 13F-HR information table for a given CIK and accession.
    
    Args:
        cik: 10-digit CIK (e.g., '0001067983')
        accession: Accession number (e.g., '0001193125-26-352200')
    
    Returns:
        DataFrame with columns: nameOfIssuer, titleOfClass, cusip, value, sshPrnamt, 
        investmentDiscretion, otherManager, votingAuthority_Sole, votingAuthority_Shared, votingAuthority_None
    """
    base_url = f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace("-", "")}/'
    
    # The information table is typically in a file like 56757.xml or similar
    # We need to find it from the filing index
    index_url = base_url + f'{accession.replace("-", "")}-index.html'
    resp = requests.get(index_url, headers=HEADERS)
    
    if resp.status_code != 200:
        # Try the text index
        index_url = base_url + f'{accession.replace("-", "")}-index-headers.html'
        resp = requests.get(index_url, headers=HEADERS)
    
    # Find the information table XML file
    import re
    xml_links = re.findall(r'href=["\']([^"\']+\.xml)["\']', resp.text)
    info_table_url = None
    for link in xml_links:
        if 'primary_doc' not in link and 'index' not in link:
            info_table_url = base_url + link
            break
    
    if not info_table_url:
        # Try common naming patterns
        for pattern in ['56757.xml', 'infoTable.xml', 'informationTable.xml']:
            test_url = base_url + pattern
            test_resp = requests.get(test_url, headers=HEADERS)
            if test_resp.status_code == 200 and 'informationTable' in test_resp.text:
                info_table_url = test_url
                break
    
    if not info_table_url:
        raise ValueError(f"Could not find 13F information table for {cik}/{accession}")
    
    # Fetch and parse the XML
    resp = requests.get(info_table_url, headers=HEADERS)
    content = resp.text
    
    root = ET.fromstring(content)
    ns = {'ns': 'http://www.sec.gov/edgar/document/thirteenf/informationtable'}
    
    holdings = []
    for info_table in root.findall('ns:infoTable', ns):
        holding = {}
        for child in info_table:
            tag = child.tag.replace('{http://www.sec.gov/edgar/document/thirteenf/informationtable}', '')
            if tag == 'shrsOrPrnAmt':
                for sub in child:
                    sub_tag = sub.tag.replace('{http://www.sec.gov/edgar/document/thirteenf/informationtable}', '')
                    holding[sub_tag] = sub.text
            elif tag == 'votingAuthority':
                for sub in child:
                    sub_tag = sub.tag.replace('{http://www.sec.gov/edgar/document/thirteenf/informationtable}', '')
                    holding[f'votingAuthority_{sub_tag}'] = sub.text
            else:
                holding[tag] = child.text
        holdings.append(holding)
    
    df = pd.DataFrame(holdings)
    
    # Convert numeric columns
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['sshPrnamt'] = pd.to_numeric(df['sshPrnamt'], errors='coerce')
    
    # Add metadata
    df['filer_cik'] = cik
    df['accession'] = accession
    df['source_form'] = '13F-HR'
    
    return df


def get_13f_filings(cik: str, max_filings: int = 5) -> list:
    """Get recent 13F-HR filings for a CIK."""
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    resp = requests.get(url, headers=HEADERS)
    data = resp.json()
    
    recent = data.get('filings', {}).get('recent', {})
    forms = recent.get('form', [])
    accessions = recent.get('accessionNumber', [])
    filing_dates = recent.get('filingDate', [])
    report_dates = recent.get('reportDate', [])
    primary_docs = recent.get('primaryDocument', [])
    
    filings = []
    for i, form in enumerate(forms):
        if '13F' in form and 'HR' in form:
            filings.append({
                'form': form,
                'accession': accessions[i],
                'filing_date': filing_dates[i],
                'report_date': report_dates[i],
                'primary_document': primary_docs[i]
            })
            if len(filings) >= max_filings:
                break
    
    return filings


def fetch_latest_13f_for_ticker(ticker: str, cik_map: dict, max_filings: int = 2) -> pd.DataFrame:
    """Fetch latest 13F-HR holdings for a ticker."""
    cik = cik_map.get(ticker.upper())
    if not cik:
        return pd.DataFrame()
    
    filings = get_13f_filings(cik, max_filings)
    all_holdings = []
    
    for filing in filings:
        try:
            print(f"  Fetching 13F-HR {filing['accession']} for {ticker}...")
            holdings = fetch_13f_holdings(cik, filing['accession'])
            holdings['filer_ticker'] = ticker
            holdings['as_of_date'] = filing['report_date']
            holdings['filing_date'] = filing['filing_date']
            all_holdings.append(holdings)
            time.sleep(0.1)  # Rate limit
        except Exception as e:
            print(f"  Error fetching {filing['accession']}: {e}")
    
    if all_holdings:
        return pd.concat(all_holdings, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    # Test with BRK-B
    with open("cik_ticker_map.json") as f:
        ticker_to_cik = {k.upper(): str(v).zfill(10) for k, v in json.load(f).items()}
    
    df = fetch_latest_13f_for_ticker('BRK-B', ticker_to_cik, max_filings=1)
    print(f"Total holdings: {len(df)}")
    if len(df) > 0:
        agg = df.groupby(['nameOfIssuer', 'titleOfClass', 'cusip']).agg({
            'value': 'sum',
            'sshPrnamt': 'sum'
        }).reset_index().sort_values('value', ascending=False)
        print(f"Unique issuers: {len(agg)}")
        print(agg.head(15).to_string())