#!/usr/bin/env python3
"""
Backfill Price History & Fundamentals for All Universe Tickers

Priority:
1. EDGAR companyfacts (XBRL) for fundamentals - preferred source
2. yfinance for prices (only practical source for 9808 tickers)
3. yfinance fundamentals as fallback for tickers without EDGAR data
4. Special handling for BRK-A/BRK-B (same CIK, different share classes)

Strategy:
- Batch process in chunks to avoid rate limits
- Resume from checkpoints
- Track success/failure per ticker
- Merge with existing parquet files (DATE-native)
"""

import json
import pandas as pd
from analytics_common import atomic_write_parquet
import numpy as np
from pathlib import Path
import yfinance as yf
import requests
import time
from datetime import datetime, date
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
CIK_MAP_FILE = 'cik_ticker_map.json'
PRICES_FILE = 'daily_prices.parquet'
FUNDAMENTALS_FILE = 'fundamentals.parquet'
CHECKPOINT_DIR = Path('backfill_checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Rate limiting
YFINANCE_DELAY = 0.1  # 10 req/sec
EDGAR_DELAY = 0.11    # 10 req/sec SEC limit

# SEC API
SEC_COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_USER_AGENT = "personal-research derek.moore@example.com"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": SEC_USER_AGENT})

# Key fundamental fields to extract from EDGAR (using standard tags)
EDGAR_TAGS = {
    # Income Statement
    'revenue_quarterly': ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet'],
    'operating_income_quarterly': ['OperatingIncomeLoss', 'IncomeFromOperations'],
    'net_income_quarterly': ['NetIncomeLoss', 'ProfitLoss'],
    'ebit': ['EarningsBeforeInterestAndTaxes', 'OperatingIncomeLoss'],
    
    # Balance Sheet
    'total_assets': ['Assets'],
    'total_liabilities': ['Liabilities'],
    'shareholders_equity': ['StockholdersEquity', 'CommonStockholdersEquity'],
    'total_debt': ['LongTermDebt', 'LongTermDebtAndCapitalLeaseObligations', 'DebtCurrent'],
    'cash_and_equivalents': ['CashAndCashEquivalentsAtCarryingValue', 'Cash'],
    'shares_outstanding': ['CommonStockSharesOutstanding', 'WeightedAverageNumberOfSharesOutstandingBasic'],
    
    # Cash Flow
    'free_cash_flow': ['FreeCashFlow', 'NetCashProvidedByUsedInOperatingActivities'],
    'capital_expenditure_ttm': ['PaymentsToAcquirePropertyPlantAndEquipment', 'CapitalExpenditures'],
    
    # Derived (computed from above)
}

# Map our fundamental column names to EDGAR tags
FUNDAMENTAL_COLUMNS = [
    'revenue_quarterly', 'operating_income_quarterly', 'net_income_quarterly', 'ebit',
    'total_assets', 'total_liabilities', 'shareholders_equity', 'total_debt',
    'cash_and_equivalents', 'shares_outstanding',
    'free_cash_flow', 'capital_expenditure_ttm',
    # Computed
    'market_cap', 'ev_ebitda', 'roic', 'roe', 'fcf_margin',
    'debt_to_equity', 'interest_coverage', 'pb_ratio', 'reinvestment_rate'
]

def load_cik_map():
    with open(CIK_MAP_FILE) as f:
        return json.load(f)

def save_checkpoint(name, data):
    with open(CHECKPOINT_DIR / f"{name}.json", 'w') as f:
        json.dump(data, f)

def load_checkpoint(name):
    path = CHECKPOINT_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def rate_limit_edgar():
    time.sleep(EDGAR_DELAY)

def rate_limit_yfinance():
    time.sleep(YFINANCE_DELAY)

def fetch_edgar_companyfacts(cik):
    """Fetch companyfacts from SEC EDGAR"""
    cik_padded = cik.zfill(10)
    url = SEC_COMPANY_FACTS.format(cik=cik_padded)
    
    rate_limit_edgar()
    try:
        resp = SESSION.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None
        else:
            print(f"  EDGAR error {resp.status_code} for CIK {cik}")
            return None
    except Exception as e:
        print(f"  EDGAR exception for CIK {cik}: {e}")
        return None

def extract_fundamentals_from_edgar(facts_json, ticker):
    """Extract quarterly fundamentals from EDGAR companyfacts"""
    if not facts_json:
        return None
    
    # EDGAR data structure: facts -> us-gaap/ifrs -> tag -> units -> USD/shares -> [values]
    facts = facts_json.get('facts', {})
    us_gaap = facts.get('us-gaap', {})
    ifrs = facts.get('ifrs', {})
    all_tags = {**us_gaap, **ifrs}
    
    # Find all quarterly dates available
    quarterly_data = {}
    
    for our_field, edgar_tags in EDGAR_TAGS.items():
        for tag in edgar_tags:
            if tag in all_tags:
                tag_data = all_tags[tag]
                units = tag_data.get('units', {})
                # Try USD first, then shares
                for unit in ['USD', 'shares']:
                    if unit in units:
                        for entry in units[unit]:
                            # Only quarterly (10-Q) and annual (10-K)
                            if entry.get('form') in ['10-Q', '10-K', '10-Q/A', '10-K/A']:
                                end_date = entry.get('end')
                                val = entry.get('val')
                                if end_date and val is not None:
                                    if end_date not in quarterly_data:
                                        quarterly_data[end_date] = {}
                                    # Handle duplicates: prefer 10-K over 10-Q, latest filing
                                    form = entry.get('form', '')
                                    if our_field not in quarterly_data[end_date] or \
                                       (form in ['10-K', '10-K/A'] and quarterly_data[end_date].get('_form', '') in ['10-Q', '10-Q/A']):
                                        quarterly_data[end_date][our_field] = val
                                        quarterly_data[end_date]['_form'] = form
                        break  # Found data for this tag
                if any(our_field in d for d in quarterly_data.values()):
                    break  # Got this field, move to next
    
    if not quarterly_data:
        return None
    
    # Convert to DataFrame
    rows = []
    for end_date, fields in quarterly_data.items():
        # Remove the _form helper
        fields_clean = {k: v for k, v in fields.items() if k != '_form'}
        row = {'ticker': ticker, 'as_of_date': end_date}
        row.update(fields_clean)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df['as_of_date'] = pd.to_datetime(df['as_of_date']).dt.date
    df = df.sort_values('as_of_date')
    
    # Deduplicate by date (keep latest form)
    df = df.drop_duplicates(subset=['ticker', 'as_of_date'], keep='last')
    
    return df

def compute_derived_fundamentals(df):
    """Compute derived metrics from base fundamentals"""
    df = df.copy()
    
    # Ensure no duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Market cap (will be filled from prices later)
    # For now, compute what we can
    
    # ROIC = EBIT * (1 - tax_rate) / (Total Assets - Current Liabilities - Cash)
    # Simplified: EBIT / (Total Assets - Cash)
    if 'ebit' in df.columns and 'total_assets' in df.columns and 'cash_and_equivalents' in df.columns:
        invested_capital = df['total_assets'] - df['cash_and_equivalents']
        df['roic'] = np.where(invested_capital > 0, df['ebit'] / invested_capital, np.nan)
    
    # ROE = Net Income / Shareholders Equity
    # ROE = TTM net income / shareholders_equity; a quarterly numerator over
    # full-year equity would understate ROE ~4x.
    if 'net_income_ttm' in df.columns and 'shareholders_equity' in df.columns:
        df['roe'] = np.where(df['shareholders_equity'] > 0, df['net_income_ttm'] / df['shareholders_equity'], np.nan)

    # FCF Margin = free_cash_flow / revenue_ttm -- free_cash_flow is TTM, so the
    # denominator is TTM revenue.
    if 'free_cash_flow' in df.columns and 'revenue_ttm' in df.columns:
        df['fcf_margin'] = np.where(df['revenue_ttm'] > 0, df['free_cash_flow'] / df['revenue_ttm'], np.nan)
    
    # Debt to Equity
    if 'total_debt' in df.columns and 'shareholders_equity' in df.columns:
        df['debt_to_equity'] = np.where(df['shareholders_equity'] > 0, df['total_debt'] / df['shareholders_equity'], np.nan)
    
    # Interest Coverage = EBIT / Interest Expense (approximate)
    # We don't have interest expense directly, skip for now
    
    # Reinvestment Rate = (CapEx - Depreciation + Change in WC) / EBIT
    # Simplified: CapEx / EBIT
    if 'capital_expenditure_ttm' in df.columns and 'ebit' in df.columns:
        df['reinvestment_rate'] = np.where(df['ebit'] > 0, df['capital_expenditure_ttm'] / df['ebit'], np.nan)
    
    return df

def fetch_yfinance_prices(ticker, start_date='1990-01-01'):
    """Fetch full price history from yfinance"""
    rate_limit_yfinance()
    try:
        tkr = yf.Ticker(ticker)
        hist = tkr.history(start=start_date, auto_adjust=True, actions=False)
        if len(hist) == 0:
            return None
        
        df = hist.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Dividends': 'dividends',
            'Stock Splits': 'stock_splits'
        })
        df['ticker'] = ticker
        df['adj_close'] = df['close']  # yfinance auto_adjust=True makes close = adj_close
        
        # Select columns we need
        cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        df = df[cols]
        return df
    except Exception as e:
        return None

def fetch_yfinance_fundamentals(ticker):
    """Fetch quarterly fundamentals from yfinance as fallback"""
    rate_limit_yfinance()
    try:
        tkr = yf.Ticker(ticker)
        
        # Get quarterly financials
        q_financials = tkr.quarterly_financials
        q_balance = tkr.quarterly_balance_sheet
        q_cashflow = tkr.quarterly_cashflow
        
        if q_financials is None or q_financials.empty:
            return None
        
        # Convert to long format
        all_data = {}
        
        # Financials (income statement)
        for idx in q_financials.index:
            for col in q_financials.columns:
                date_key = pd.to_datetime(col).date()
                if date_key not in all_data:
                    all_data[date_key] = {}
                all_data[date_key][idx.lower().replace(' ', '_')] = q_financials.loc[idx, col]
        
        # Balance sheet
        for idx in q_balance.index:
            for col in q_balance.columns:
                date_key = pd.to_datetime(col).date()
                if date_key not in all_data:
                    all_data[date_key] = {}
                all_data[date_key][idx.lower().replace(' ', '_')] = q_balance.loc[idx, col]
        
        # Cash flow
        for idx in q_cashflow.index:
            for col in q_cashflow.columns:
                date_key = pd.to_datetime(col).date()
                if date_key not in all_data:
                    all_data[date_key] = {}
                all_data[date_key][idx.lower().replace(' ', '_')] = q_cashflow.loc[idx, col]
        
        # Convert to DataFrame
        rows = []
        for date_key, fields in all_data.items():
            row = {'ticker': ticker, 'as_of_date': date_key}
            row.update(fields)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df['as_of_date'] = pd.to_datetime(df['as_of_date']).dt.date
        df = df.sort_values('as_of_date')
        
        # Deduplicate by date
        df = df.drop_duplicates(subset=['ticker', 'as_of_date'], keep='last')
        
        return df
    except Exception as e:
        return None

def standardize_fundamentals(df, source='edgar'):
    """Standardize column names to our schema"""
    # Map common yfinance/EDGAR field names to our standard names
    column_map = {
        # Income statement
        'revenue_quarterly': 'revenue_quarterly',
        'revenue_quarterly': 'revenue_quarterly',
        'revenues': 'revenue_quarterly',
        'operating_income_quarterly': 'operating_income_quarterly',
        'operating_income_loss': 'operating_income_quarterly',
        'net_income_quarterly': 'net_income_quarterly',
        'net_income_loss': 'net_income_quarterly',
        'ebit': 'ebit',
        'earnings_before_interest_and_taxes': 'ebit',
        
        # Balance sheet
        'total_assets': 'total_assets',
        'total_liabilities': 'total_liabilities',
        'shareholders_equity': 'shareholders_equity',
        'common_stockholders_equity': 'shareholders_equity',
        'total_debt': 'total_debt',
        'long_term_debt': 'total_debt',
        'cash_and_cash_equivalents_at_carrying_value': 'cash_and_equivalents',
        'cash_and_cash_equivalents': 'cash_and_equivalents',
        'common_stock_shares_outstanding': 'shares_outstanding',
        'weighted_average_number_of_shares_outstanding_basic': 'shares_outstanding',
        
        # Cash flow
        'free_cash_flow': 'free_cash_flow',
        'net_cash_provided_by_used_in_operating_activities': 'free_cash_flow',
        'payments_to_acquire_property_plant_and_equipment': 'capital_expenditure_ttm',
        'capital_expenditures': 'capital_expenditure_ttm',
    }
    
    # Rename columns
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    
    # Keep only our standard columns (plus ticker, as_of_date)
    keep_cols = ['ticker', 'as_of_date'] + FUNDAMENTAL_COLUMNS
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols]
    
    return df

def merge_fundamentals(existing, new, source_priority='edgar'):
    """Merge new fundamentals with existing, preferring EDGAR"""
    if existing is None or len(existing) == 0:
        return new
    if new is None or len(new) == 0:
        return existing
    
    # Combine and deduplicate by ticker + as_of_date
    # Priority: EDGAR > yfinance
    combined = pd.concat([existing, new], ignore_index=True)
    
    # Sort by source priority (EDGAR first)
    # We need a source column - for now, just drop duplicates keeping first
    combined = combined.sort_values(['ticker', 'as_of_date'])
    combined = combined.drop_duplicates(subset=['ticker', 'as_of_date'], keep='first')
    
    return combined

def merge_prices(existing, new):
    """Merge new prices with existing"""
    if existing is None or len(existing) == 0:
        return new
    if new is None or len(new) == 0:
        return existing
    
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.sort_values(['ticker', 'date'])
    combined = combined.drop_duplicates(subset=['ticker', 'date'], keep='last')
    return combined

def main():
    print("=== BACKFILL PRICE HISTORY & FUNDAMENTALS ===\n")
    
    # Load current data
    print("Loading existing data...")
    cik_map = load_cik_map()
    print(f"CIK map: {len(cik_map)} tickers")
    
    try:
        existing_prices = pd.read_parquet(PRICES_FILE)
        existing_prices['date'] = pd.to_datetime(existing_prices['date']).dt.date
        print(f"Existing prices: {len(existing_prices):,} rows, {existing_prices['ticker'].nunique()} tickers")
    except:
        existing_prices = None
        print("Existing prices: None")
    
    try:
        existing_fund = pd.read_parquet(FUNDAMENTALS_FILE)
        existing_fund['as_of_date'] = pd.to_datetime(existing_fund['as_of_date']).dt.date
        print(f"Existing fundamentals: {len(existing_fund):,} rows, {existing_fund['ticker'].nunique()} tickers")
    except:
        existing_fund = None
        print("Existing fundamentals: None")
    
    # Identify missing tickers
    price_tickers = set(existing_prices['ticker'].unique()) if existing_prices is not None else set()
    fund_tickers = set(existing_fund['ticker'].unique()) if existing_fund is not None else set()
    
    all_tickers = list(cik_map.keys())
    missing_price_tickers = [t for t in all_tickers if t not in price_tickers]
    missing_fund_tickers = [t for t in all_tickers if t not in fund_tickers]
    
    print(f"\nMissing prices: {len(missing_price_tickers)}")
    print(f"Missing fundamentals: {len(missing_fund_tickers)}")
    
    # Load checkpoints
    price_checkpoint = load_checkpoint('price_backfill')
    fund_checkpoint = load_checkpoint('fund_backfill')
    
    done_price = set(price_checkpoint.get('done', [])) if price_checkpoint else set()
    done_fund = set(fund_checkpoint.get('done', [])) if fund_checkpoint else set()
    failed_price = set(price_checkpoint.get('failed', [])) if price_checkpoint else set()
    failed_fund = set(fund_checkpoint.get('failed', [])) if fund_checkpoint else set()
    
    # Filter out already done
    todo_price = [t for t in missing_price_tickers if t not in done_price and t not in failed_price]
    todo_fund = [t for t in missing_fund_tickers if t not in done_fund and t not in failed_fund]
    
    print(f"TODO prices: {len(todo_price)} (done: {len(done_price)}, failed: {len(failed_price)})")
    print(f"TODO fundamentals: {len(todo_fund)} (done: {len(done_fund)}, failed: {len(failed_fund)})")
    
    # Process in batches
    BATCH_SIZE = 50
    
    # ===== FUNDAMENTALS FIRST (EDGAR preferred) =====
    print(f"\n{'='*60}")
    print("PHASE 1: FUNDAMENTALS BACKFILL (EDGAR preferred)")
    print(f"{'='*60}")
    
    for i in range(0, len(todo_fund), BATCH_SIZE):
        batch = todo_fund[i:i+BATCH_SIZE]
        print(f"\nBatch {i//BATCH_SIZE + 1}/{(len(todo_fund)-1)//BATCH_SIZE + 1}: {len(batch)} tickers")
        
        batch_fundamentals = []
        
        for ticker in tqdm(batch, desc="Fundamentals"):
            cik = cik_map[ticker]
            
            # Try EDGAR first
            edgar_data = fetch_edgar_companyfacts(cik)
            if edgar_data:
                fund_df = extract_fundamentals_from_edgar(edgar_data, ticker)
                if fund_df is not None and len(fund_df) > 0:
                    fund_df = compute_derived_fundamentals(fund_df)
                    fund_df['source'] = 'edgar'
                    batch_fundamentals.append(fund_df)
                    done_fund.add(ticker)
                    continue
            
            # Fallback to yfinance
            yf_data = fetch_yfinance_fundamentals(ticker)
            if yf_data is not None and len(yf_data) > 0:
                yf_data = standardize_fundamentals(yf_data, 'yfinance')
                yf_data = compute_derived_fundamentals(yf_data)
                yf_data['source'] = 'yfinance'
                batch_fundamentals.append(yf_data)
                done_fund.add(ticker)
            else:
                failed_fund.add(ticker)
        
        # Merge batch into existing
        if batch_fundamentals:
            batch_df = pd.concat(batch_fundamentals, ignore_index=True)
            existing_fund = merge_fundamentals(existing_fund, batch_df)
            print(f"  Merged {len(batch_df)} rows, total: {len(existing_fund):,}")
        
        # Save checkpoint
        save_checkpoint('fund_backfill', {'done': list(done_fund), 'failed': list(failed_fund)})
        
        # Save progress every 5 batches
        if (i // BATCH_SIZE) % 5 == 0:
            atomic_write_parquet(existing_fund, FUNDAMENTALS_FILE)
            print(f"  Saved progress to {FUNDAMENTALS_FILE}")
    
    # Final save
    atomic_write_parquet(existing_fund, FUNDAMENTALS_FILE)
    print(f"\nFundamentals saved: {len(existing_fund):,} rows")
    
    # ===== PRICES =====
    print(f"\n{'='*60}")
    print("PHASE 2: PRICE HISTORY BACKFILL (yfinance)")
    print(f"{'='*60}")
    
    for i in range(0, len(todo_price), BATCH_SIZE):
        batch = todo_price[i:i+BATCH_SIZE]
        print(f"\nBatch {i//BATCH_SIZE + 1}/{(len(todo_price)-1)//BATCH_SIZE + 1}: {len(batch)} tickers")
        
        batch_prices = []
        
        for ticker in tqdm(batch, desc="Prices"):
            price_df = fetch_yfinance_prices(ticker)
            if price_df is not None and len(price_df) > 0:
                batch_prices.append(price_df)
                done_price.add(ticker)
            else:
                failed_price.add(ticker)
        
        if batch_prices:
            batch_df = pd.concat(batch_prices, ignore_index=True)
            existing_prices = merge_prices(existing_prices, batch_df)
            print(f"  Merged {len(batch_df)} rows, total: {len(existing_prices):,}")
        
        save_checkpoint('price_backfill', {'done': list(done_price), 'failed': list(failed_price)})
        
        if (i // BATCH_SIZE) % 5 == 0:
            existing_prices.to_parquet(PRICES_FILE, index=False)
            print(f"  Saved progress to {PRICES_FILE}")
    
    # Final save
    existing_prices.to_parquet(PRICES_FILE, index=False)
    print(f"\nPrices saved: {len(existing_prices):,} rows")
    
    # ===== BRK-A SPECIAL HANDLING =====
    print(f"\n{'='*60}")
    print("PHASE 3: BRK-A / BRK-B SPECIAL HANDLING")
    print(f"{'='*60}")
    
    # BRK-A and BRK-B share CIK 0001067983
    # BRK-B has prices, BRK-A doesn't
    # BRK-B has fundamentals (from yfinance), need EDGAR for both
    
    # Fetch EDGAR for BRK CIK
    brk_cik = cik_map.get('BRK-B') or cik_map.get('BRK-A')
    if brk_cik:
        print(f"Fetching EDGAR for BRK (CIK: {brk_cik})...")
        edgar_data = fetch_edgar_companyfacts(brk_cik)
        if edgar_data:
            # Extract for both tickers (same fundamentals, different share counts)
            for ticker in ['BRK-A', 'BRK-B']:
                fund_df = extract_fundamentals_from_edgar(edgar_data, ticker)
                if fund_df is not None and len(fund_df) > 0:
                    fund_df = compute_derived_fundamentals(fund_df)
                    fund_df['source'] = 'edgar'
                    existing_fund = merge_fundamentals(existing_fund, fund_df)
                    print(f"  Added EDGAR fundamentals for {ticker}: {len(fund_df)} rows")
    
    # BRK-A price: derive from BRK-B * 1500 (approximate)
    if 'BRK-B' in (existing_prices['ticker'].unique() if existing_prices is not None else []):
        brk_b_prices = existing_prices[existing_prices['ticker'] == 'BRK-B'].copy()
        if len(brk_b_prices) > 0:
            brk_a_prices = brk_b_prices.copy()
            brk_a_prices['ticker'] = 'BRK-A'
            brk_a_prices['adj_close'] = brk_a_prices['adj_close'] * 1500
            brk_a_prices['close'] = brk_a_prices['close'] * 1500
            brk_a_prices['open'] = brk_a_prices['open'] * 1500
            brk_a_prices['high'] = brk_a_prices['high'] * 1500
            brk_a_prices['low'] = brk_a_prices['low'] * 1500
            existing_prices = merge_prices(existing_prices, brk_a_prices)
            print(f"  Derived BRK-A prices from BRK-B: {len(brk_a_prices)} rows")
    
    # Final save
    atomic_write_parquet(existing_fund, FUNDAMENTALS_FILE)
    existing_prices.to_parquet(PRICES_FILE, index=False)
    
    print(f"\n{'='*60}")
    print("BACKFILL COMPLETE")
    print(f"{'='*60}")
    print(f"Prices: {len(existing_prices):,} rows, {existing_prices['ticker'].nunique()} tickers")
    print(f"Fundamentals: {len(existing_fund):,} rows, {existing_fund['ticker'].nunique()} tickers")
    print(f"Price failures: {len(failed_price)}")
    print(f"Fund failures: {len(failed_fund)}")

if __name__ == '__main__':
    main()