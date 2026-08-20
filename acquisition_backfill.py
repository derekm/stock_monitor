#!/usr/bin/env python3
"""
acquisition_backfill.py — Detect corporate actions and trigger existing
backfill/ingest pipelines. Does NOT reimplement retrieval — uses:
  - backfill_edgar.py : SEC companyfacts → fundamentals.parquet
  - update_polygon.py : Polygon bulk API → daily_prices.parquet (or yfinance fallback)
  - lookthrough_engine.py : pro forma combination during acquisition window

Universe = daily_prices.parquet (NOT monitored_stocks).
"""

import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
FUND = DATA_DIR / "fundamentals.parquet"
CORPORATE_ACTIONS = DATA_DIR / "corporate_actions.parquet"
CIK_MAP = DATA_DIR / "cik_ticker_map.json"

UA = {'User-Agent': 'personal-research derek.moore@example.com'}

# ─────────────────────────────────────────────────────────────────────────────
# CORPORATE ACTION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# M&A-related XBRL tags in companyfacts
MA_TAGS = [
    'BusinessCombinationConsiderationTransferred',
    'BusinessCombinationConsiderationTransferredEquityInterestsIssuedAndIssuable',
    'BusinessAcquisitionPurchasePriceAllocationGoodwillAmount',
    'BusinessAcquisitionPurchasePriceAllocationAssetsAcquiredLiabilitiesAssumedNet',
    'BusinessCombinationRecognizedIdentifiableAssetsAcquiredGoodwillAndLiabilitiesAssumedNet',
    'BusinessCombinationContingentConsiderationLiabilityCurrent',
    'BusinessCombinationContingentConsiderationLiabilityNoncurrent',
    'PaymentsOfMergerRelatedCostsFinancingActivities',
    'StockIssuedDuringPeriodValueAcquisitions',
    'NoncashOrPartNoncashAcquisitionFixedAssetsAcquired1',
    'GoodwillPurchaseAccountingAdjustments',
]

# SEC form types that signal acquisitions
MA_FORM_TYPES = ['8-K', 'DEFM14A', 'S-4', 'S-4/A', '425', '6-K', '6-K/A']

# Keywords in filing titles/descriptions
MA_KEYWORDS = [
    'acquisition', 'acquired', 'merger', 'merge', 'business combination',
    'purchase agreement', 'definitive agreement', 'tender offer',
    'acquire', 'acquiring', 'combination', 'consolidation',
]


def load_cik_map() -> dict:
    """Load CIK-to-ticker mapping."""
    if CIK_MAP.exists():
        with open(CIK_MAP) as f:
            return json.load(f)
    return {}


def ticker_to_cik(ticker: str, cik_map: dict) -> str | None:
    """Resolve ticker to CIK."""
    if ticker in cik_map:
        return cik_map[ticker]
    if ticker.upper() in cik_map:
        return cik_map[ticker.upper()]
    return None


def cik_to_ticker(cik: str, cik_map: dict) -> str | None:
    """Reverse lookup CIK to ticker."""
    for ticker, mapped_cik in cik_map.items():
        if mapped_cik == cik:
            return ticker
    return None


def detect_acquisitions_from_companyfacts(cik: str, ticker: str) -> list[dict]:
    """
    Detect acquisitions from SEC EDGAR companyfacts API.
    Looks for M&A-related XBRL tags.
    """
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
    
    try:
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200:
            return []
        
        d = r.json()
        facts = d.get('facts', {}).get('us-gaap', {})
        
        acquisitions = []
        
        for tag in MA_TAGS:
            if tag not in facts:
                continue
            
            units = facts[tag].get('units', {}).get('USD', [])
            for entry in units:
                val = entry.get('val', 0)
                if val and val > 1_000_000:  # > $1M to filter noise
                    acquisitions.append({
                        'tag': tag,
                        'date': entry.get('end'),
                        'value': val,
                        'frame': entry.get('frame', ''),
                        'fiscal_year': entry.get('fy'),
                        'fiscal_period': entry.get('fp'),
                    })
        
        return acquisitions
    except Exception as e:
        print(f"  Error fetching companyfacts for {ticker}: {e}")
        return []


def detect_acquisitions_from_filings(cik: str, ticker: str) -> list[dict]:
    """
    Detect acquisitions from SEC EDGAR filings index.
    Looks for 8-K, DEFM14A, S-4 forms with M&A keywords.
    """
    base_url = f'https://data.sec.gov/browse-edgar/company'
    
    acquisitions = []
    
    for form_type in MA_FORM_TYPES:
        params = {
            'action': 'getcompany',
            'CIK': cik,
            'type': form_type,
            'dateb': '',
            'owner': 'include',
            'count': '20',
            'search_text': '',
        }
        
        try:
            r = requests.get(base_url, headers=UA, params=params, timeout=30)
            if r.status_code != 200:
                continue
            
            content = r.text
            
            for keyword in MA_KEYWORDS:
                if keyword.lower() in content.lower():
                    acquisitions.append({
                        'form_type': form_type,
                        'keyword': keyword,
                        'source': 'filing_index',
                        'date': None,  # Would need to parse from filing
                    })
            
            time.sleep(0.15)  # Rate limit
        except Exception as e:
            print(f"  Error fetching {form_type} for {ticker}: {e}")
            continue
    
    return acquisitions


def detect_acquisitions_for_ticker(ticker: str) -> list[dict]:
    """
    Detect all acquisitions for a ticker using multiple SEC sources.
    """
    cik_map = load_cik_map()
    cik = ticker_to_cik(ticker, cik_map)
    
    if not cik:
        print(f"  No CIK found for {ticker}")
        return []
    
    print(f"  Detecting acquisitions for {ticker} (CIK {cik})...")
    
    # Method 1: companyfacts M&A tags
    companyfacts_acqs = detect_acquisitions_from_companyfacts(cik, ticker)
    
    # Method 2: filings index
    filings_acqs = detect_acquisitions_from_filings(cik, ticker)
    
    # Combine and deduplicate
    all_acqs = companyfacts_acqs + filings_acqs
    
    print(f"  Found {len(companyfacts_acqs)} from companyfacts, {len(filings_acqs)} from filings")
    
    return all_acqs


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE CHECK (daily_prices only, NOT monitored_stocks)
# ─────────────────────────────────────────────────────────────────────────────

def get_universe_tickers() -> set[str]:
    """Get current universe from daily_prices.parquet."""
    if PRICES.exists():
        df = pd.read_parquet(PRICES, columns=['ticker'])
        return set(df['ticker'].unique())
    return set()


def ticker_in_universe(ticker: str) -> bool:
    """Check if ticker exists in daily_prices."""
    return ticker in get_universe_tickers()


def ticker_has_fundamentals(ticker: str) -> bool:
    """Check if ticker exists in fundamentals."""
    if FUND.exists():
        df = pd.read_parquet(FUND, columns=['ticker'])
        return ticker in df['ticker'].values
    return False


# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER EXISTING BACKFILL PIPELINES (no reimplementation)
# ─────────────────────────────────────────────────────────────────────────────

def backfill_target_prices(ticker: str, days: int = 365*20) -> bool:
    """
    Backfill price history for a target ticker.
    Uses update_polygon.py bulk ingest (which handles delisted tickers),
    falls back to yfinance if Polygon is unavailable.
    """
    print(f"  Backfilling prices for {ticker}...")
    
    # Primary: Polygon (handles delisted tickers)
    import subprocess
    import os
    
    polygon_key = os.environ.get("POLYGON_API_KEY", "")
    if polygon_key:
        try:
            print(f"  Pulling from Polygon (last {days} days)...")
            result = subprocess.run(
                ["python", "update_polygon.py", "--days", str(days), "--save"],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                print(f"  Polygon backfill complete for universe")
                return True
            else:
                print(f"  Polygon error: {result.stderr[:200]}")
        except Exception as e:
            print(f"  Polygon failed: {e}")
    
    # Fallback: yfinance
    try:
        import yfinance as yf
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            for suffix in ['', '.old', '-old']:
                try:
                    stock = yf.Ticker(ticker + suffix)
                    hist = stock.history(start=start_date, end=end_date)
                    if not hist.empty:
                        break
                except Exception:
                    continue
        
        if hist.empty:
            print(f"  No price data available for {ticker} (possibly delisted)")
            return False
        
        prices_df = pd.DataFrame({
            'ticker': ticker,
            'date': hist.index.date,
            'adj_close': hist['Close'].values,
            'volume': hist['Volume'].values,
        })
        
        if PRICES.exists():
            existing = pd.read_parquet(PRICES)
            mask = (existing['ticker'] == ticker) & (
                pd.to_datetime(existing['date']).dt.date.isin(set(prices_df['date']))
            )
            existing = existing[~mask]
            combined = pd.concat([existing, prices_df], ignore_index=True)
        else:
            combined = prices_df
        
        combined.to_parquet(PRICES, index=False)
        print(f"  Backfilled {len(prices_df)} price rows for {ticker} via yfinance")
        return True
        
    except Exception as e:
        print(f"  Error backfilling prices for {ticker}: {e}")
        return False


def backfill_target_fundamentals(ticker: str) -> bool:
    """
    Backfill fundamentals for a target ticker using SEC EDGAR companyfacts.
    """
    print(f"  Backfilling fundamentals for {ticker}...")
    
    cik_map = load_cik_map()
    cik = ticker_to_cik(ticker, cik_map)
    
    if not cik:
        print(f"  No CIK found for {ticker}, trying yfinance...")
        return backfill_target_fundamentals_yfinance(ticker)
    
    try:
        # Use the canonical retrieval from backfill_edgar
        from backfill_edgar import fetch_ticker, build_rows
        from analytics_common import load_adj_prices_pandas, atomic_write_parquet
        
        # Fetch from SEC
        frames = fetch_ticker(ticker, cik)
        if not frames:
            return backfill_target_fundamentals_yfinance(ticker)
        
        # Get prices for market cap calc
        prices = load_adj_prices_pandas(tickers=[ticker])
        px = {tk: g.set_index("date")["close"] for tk, g in prices.groupby("ticker")}
        
        # Build rows (additive merge handled by update_fundamentals)
        rows = build_rows(ticker, frames, px)
        
        if rows:
            new_df = pd.DataFrame(rows)
            
            # Append to existing fundamentals
            if FUND.exists():
                existing = pd.read_parquet(FUND)
                # Remove overlapping rows
                mask = (existing['ticker'] == ticker) & (existing['as_of_date'].isin(new_df['as_of_date']))
                existing = existing[~mask]
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
            
            atomic_write_parquet(combined, FUND)
            print(f"  Backfilled {len(rows)} fundamental rows for {ticker}")
            return True
        
        return backfill_target_fundamentals_yfinance(ticker)
        
    except Exception as e:
        print(f"  Error backfilling fundamentals for {ticker}: {e}")
        return backfill_target_fundamentals_yfinance(ticker)


def backfill_target_fundamentals_yfinance(ticker: str) -> bool:
    """Fallback: backfill fundamentals from yfinance."""
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            return False
        
        # Extract what we can from yfinance info
        shares = info.get('sharesOutstanding')
        market_cap = info.get('marketCap')
        
        # Get quarterly financials
        try:
            quarterly = stock.quarterly_financials
            if quarterly is not None and not quarterly.empty:
                rows = []
                for col in quarterly.columns:
                    date = col.date() if hasattr(col, 'date') else col
                    row_data = quarterly[col]
                    
                    rows.append({
                        'ticker': ticker,
                        'as_of_date': date,
                        'shares_outstanding': shares,
                        'market_cap': market_cap,
                        'source': 'yfinance',
                    })
                
                if rows:
                    new_df = pd.DataFrame(rows)
                    
                    if FUND.exists():
                        existing = pd.read_parquet(FUND)
                        mask = (existing['ticker'] == ticker) & (existing['as_of_date'].isin(new_df['as_of_date']))
                        existing = existing[~mask]
                        combined = pd.concat([existing, new_df], ignore_index=True)
                    else:
                        combined = new_df
                    
                    atomic_write_parquet(combined, FUND)
                    print(f"  Backfilled {len(rows)} fundamental rows from yfinance for {ticker}")
                    return True
        except Exception:
            pass
        
        return False
        
    except Exception as e:
        print(f"  Error with yfinance for {ticker}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_acquisition(acquirer_ticker: str, target_ticker: str,
                        completion_date: str, announcement_date: str = None):
    """
    Full pipeline for processing an acquisition:
    1. Check if target exists in universe (daily_prices)
    2. If missing: backfill prices + fundamentals using existing pipelines
    3. Register in corporate_actions
    4. Enable look-through
    """
    print(f"\nProcessing acquisition: {acquirer_ticker} acquires {target_ticker}")
    print(f"  Completion: {completion_date}")
    
    # Step 1: Check if target exists in universe
    in_prices = ticker_in_universe(target_ticker)
    has_fundamentals = ticker_has_fundamentals(target_ticker)
    print(f"  Target status: prices={in_prices}, fundamentals={has_fundamentals}")
    
    # Step 2: Backfill if missing
    price_success = False
    fund_success = False
    
    if not in_prices:
        print(f"  Backfilling prices for {target_ticker}...")
        price_success = backfill_target_prices(target_ticker)
    
    if not has_fundamentals:
        print(f"  Backfilling fundamentals for {target_ticker}...")
        fund_success = backfill_target_fundamentals(target_ticker)
    else:
        fund_success = True  # Already have it
    
    if (not in_prices and price_success) or (not has_fundamentals and fund_success) or (in_prices and has_fundamentals):
        if price_success or fund_success:
            print(f"  Backfill complete: prices={price_success}, fundamentals={fund_success}")
        else:
            print(f"  {target_ticker} already fully in universe")
    else:
        print(f"  WARNING: Could not fully backfill {target_ticker} (prices={price_success}, fundamentals={fund_success})")
    
    # Step 3: Register in corporate_actions
    from lookthrough_engine import add_acquisition
    add_acquisition(
        acquirer_ticker=acquirer_ticker,
        target_ticker=target_ticker,
        completion_date=completion_date,
        announcement_date=announcement_date,
    )
    
    print(f"  Acquisition registered: {acquirer_ticker} + {target_ticker}")


def auto_detect_and_backfill(tickers: list[str] = None):
    """
    Auto-detect acquisitions for a list of tickers and backfill missing targets.
    Universe = daily_prices (not monitored_stocks).
    """
    if tickers is None:
        # Use universe from daily_prices
        tickers = sorted(get_universe_tickers())
    
    print(f"Scanning {len(tickers)} tickers for acquisitions...")
    
    total_acquisitions = 0
    
    for i, ticker in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}] {ticker}")
        
        acquisitions = detect_acquisitions_for_ticker(ticker)
        
        if acquisitions:
            total_acquisitions += len(acquisitions)
            print(f"  Found {len(acquisitions)} potential acquisitions")
            # Note: resolving target tickers from acquisition details requires
            # full-text parsing of filings. For now, manual registration via
            # process_acquisition() is the reliable path.
        
        # Rate limit
        time.sleep(0.15)
    
    print(f"\nScan complete: {total_acquisitions} acquisitions detected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to scan")
    parser.add_argument("--process", nargs=4, metavar=('ACQUIRER', 'TARGET', 'CLOSE_DATE', 'ANNOUNCE_DATE'),
                       help="Manually register an acquisition: ACQUIRER TARGET CLOSE_DATE ANNOUNCE_DATE")
    args = parser.parse_args()
    
    if args.process:
        acquirer, target, close_date, announce_date = args.process
        process_acquisition(acquirer, target, close_date, announce_date)
    elif args.tickers:
        auto_detect_and_backfill(args.tickers)
    else:
        # Default: scan universe
        auto_detect_and_backfill()