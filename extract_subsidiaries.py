#!/usr/bin/env python3
"""
Extract corporate subsidiary data from SEC Exhibit 21.1 filings.

Exhibit 21.1 = "Subsidiaries of the Registrant" - required in 10-K filings.
Lists all subsidiaries with jurisdiction of incorporation.
"""

import pandas as pd
import requests
import re
import json
import time
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup

# Load CIK map
with open("cik_ticker_map.json") as f:
    CIK_MAP = {k.upper(): str(v).zfill(10) for k, v in json.load(f).items()}

HEADERS = {'User-Agent': 'personal-research derek.moore@example.com'}
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

def get_latest_10k(cik: str) -> str:
    """Get latest 10-K accession for a CIK (filed by company, not agent)"""
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        recent = data['filings']['recent']
        forms = recent['form']
        accessions = recent['accessionNumber']
        
        # Prefer 10-K filed by the company itself (accession starts with CIK prefix)
        cik_prefix = cik[:5]  # First 5 digits usually enough
        for f, a in zip(forms, accessions):
            if f == '10-K' and a.startswith(cik_prefix):
                return a
        
        # Fallback: any 10-K
        for f, a in zip(forms, accessions):
            if f == '10-K':
                return a
    except Exception as e:
        print(f"  Error getting 10-K for {cik}: {e}")
    return None

def find_exhibit_21_url(cik: str, accession: str) -> str:
    """Find Exhibit 21.1 URL in 10-K filing directory"""
    base_url = f"{SEC_ARCHIVES}/{int(cik)}/{accession.replace('-', '')}/"
    
    # Get directory listing
    resp = requests.get(base_url, headers=HEADERS)
    if resp.status_code != 200:
        return None
    
    # Look for exhibit 21 file - various patterns
    import re
    patterns = [
        r'href="([^"]*exhibit21[^"]+\.htm)"',
        r'href="([^"]*ex-21[^"]+\.htm)"',
        r'href="([^"]*ex21[^"]+\.htm)"',
        r'href="([^"]*exhibit-21[^"]+\.htm)"',
    ]
    
    for line in resp.text.split('\n'):
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                link = match.group(1)
                # If it's already an absolute URL, return as-is
                if link.startswith('http'):
                    return link
                # If it starts with /, make it absolute
                if link.startswith('/'):
                    return f"https://www.sec.gov{link}"
                # Otherwise relative to base_url
                return base_url + link
    
    # Fallback: look for any file with '21' in name and 'ex' prefix
    all_links = re.findall(r'href="([^"]+\.htm)"', resp.text)
    for link in all_links:
        if '21' in link and ('ex' in link.lower() or 'exhibit' in link.lower()):
            if not any(x in link for x in ['index', 'header']):
                if link.startswith('http'):
                    return link
                if link.startswith('/'):
                    return f"https://www.sec.gov{link}"
                return base_url + link
    
    return None

def parse_exhibit_21(html_content: str) -> list:
    """Parse Exhibit 21.1 HTML table for subsidiaries"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    subsidiaries = []
    
    # Find tables with subsidiary data
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            
            # Look for rows with subsidiary name and jurisdiction
            # MSFT format: Name, (empty), Where Incorporated
            if len(cell_texts) >= 3:
                name = cell_texts[0]
                jurisdiction = cell_texts[2] if len(cell_texts) > 2 else cell_texts[-1]
                
                # Skip headers and empty rows
                if (name and jurisdiction and 
                    name.lower() not in ['name', 'subsidiary', 'where incorporated', 'jurisdiction'] and
                    len(name) > 2 and len(jurisdiction) > 2):
                    subsidiaries.append({
                        'subsidiary_name': name,
                        'jurisdiction': jurisdiction
                    })
    
    return subsidiaries

def extract_subsidiaries_for_ticker(ticker: str) -> list:
    """Extract all subsidiaries for one ticker"""
    cik = CIK_MAP.get(ticker.upper())
    if not cik:
        return []
    
    accession = get_latest_10k(cik)
    if not accession:
        return []
    
    exhibit_url = find_exhibit_21_url(cik, accession)
    if not exhibit_url:
        return []
    
    resp = requests.get(exhibit_url, headers=HEADERS)
    if resp.status_code != 200:
        return []
    
    subs = parse_exhibit_21(resp.text)
    
    # Add metadata
    for s in subs:
        s['parent_ticker'] = ticker
        s['parent_cik'] = cik
        s['as_of_date'] = accession.split('-')[1] + '-' + accession.split('-')[2]  # approximate
    
    return subs

def main():
    # Test with known companies
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B', 'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK']
    
    all_subs = []
    for ticker in tqdm(test_tickers, desc="Extracting subsidiaries"):
        try:
            subs = extract_subsidiaries_for_ticker(ticker)
            if subs:
                all_subs.extend(subs)
                print(f"  {ticker}: {len(subs)} subsidiaries")
            else:
                print(f"  {ticker}: No subsidiaries found")
            time.sleep(0.5)
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")
    
    if all_subs:
        df = pd.DataFrame(all_subs)
        df.to_parquet('corporate_subsidiaries.parquet', index=False)
        print(f"\nSaved {len(df)} subsidiaries to corporate_subsidiaries.parquet")
        print(df.head(20).to_string())
    else:
        print("No subsidiaries extracted!")

if __name__ == "__main__":
    main()