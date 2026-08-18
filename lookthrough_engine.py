#!/usr/bin/env python3
"""
lookthrough_engine.py — Generalized Pro Forma Financial Combination for Acquisitions.

Handles:
- Multiple acquisitions in same period (overlapping look-through windows)
- Two provenance columns: data_provenance + lookthrough_source
- Generalized for all companies, not ticker-specific

Provenance Columns:
  data_provenance: "standalone" | "lookthrough_proforma"
  lookthrough_source: comma-separated list of combined tickers, or None for standalone
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
CORPORATE_ACTIONS = DATA_DIR / "corporate_actions.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# ACQUISITION REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

def load_acquisitions() -> pd.DataFrame:
    """Load acquisition records from corporate_actions."""
    if not CORPORATE_ACTIONS.exists():
        return pd.DataFrame()
    ca = pd.read_parquet(CORPORATE_ACTIONS)
    mask = ca['action_type'].isin(['acquisition', 'merger'])
    return ca[mask].copy()


def get_acquisitions_for(ticker: str) -> pd.DataFrame:
    """
    Get all acquisitions where `ticker` is the acquirer.
    Returns DataFrame sorted by completion_date.
    """
    acqs = load_acquisitions()
    if acqs.empty:
        return acqs
    
    mask = acqs['acquirer_ticker'] == ticker
    return acqs[mask].sort_values('completion_date').copy()


def get_acquisitions_during(ticker: str, start_date, end_date) -> pd.DataFrame:
    """
    Get acquisitions active during a period.
    An acquisition is "active" for look-through from announcement/completion
    until the acquirer reports actual combined results (next quarter after completion).
    """
    acqs = get_acquisitions_for(ticker)
    if acqs.empty:
        return acqs
    
    # Filter to acquisitions whose look-through window overlaps [start_date, end_date]
    # Look-through window: [announcement_date, completion_date + 1 quarter]
    acqs['lookthrough_start'] = pd.to_datetime(acqs['announcement_date']).dt.date
    acqs['lookthrough_end'] = pd.to_datetime(acqs['completion_date']).dt.date + pd.DateOffset(months=3)
    
    mask = (acqs['lookthrough_start'] <= end_date) & (acqs['lookthrough_end'] >= start_date)
    return acqs[mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# PRO FORMA COMBINATION
# ─────────────────────────────────────────────────────────────────────────────

def combine_quarterly_rows(
    acquirer_row: pd.Series,
    target_rows: dict[str, pd.Series] | pd.Series,
) -> pd.Series:
    """
    Combine acquirer quarterly fundamentals with one or more target rows.
    
    Args:
        acquirer_row: Single row of acquirer fundamentals
        target_rows: Dict of {ticker: row} for multiple acquisitions, or single Series
    
    Returns:
        Combined row with updated provenance columns
    """
    combined = acquirer_row.copy()
    
    # Normalize target_rows to dict
    if isinstance(target_rows, pd.Series):
        target_rows = {'unknown': target_rows}
    
    # Income statement items (additive)
    additive_cols = [
        'total_revenue', 'operating_income', 'net_income',
        'free_cash_flow', 'total_assets', 'total_debt',
        'shareholders_equity', 'cash_and_equivalents',
        'total_liabilities', 'capital_expenditure',
        'ttm_revenue', 'ttm_net_income', 'ttm_operating_income',
        'ttm_operating_cash_flow', 'ttm_capital_expenditure',
    ]
    
    for col in additive_cols:
        if col not in acquirer_row.index:
            continue
        a_val = acquirer_row[col]
        if pd.isna(a_val):
            a_val = 0.0
        
        total = a_val
        for ticker, t_row in target_rows.items():
            t_val = t_row.get(col, np.nan)
            if pd.notna(t_val):
                total += t_val
        
        combined[col] = total
    
    # Recompute ratios from combined values
    rev = combined.get('total_revenue', np.nan)
    fcf = combined.get('free_cash_flow', np.nan)
    if pd.notna(rev) and rev > 0 and pd.notna(fcf):
        combined['fcf_margin'] = fcf / rev
    
    equity = combined.get('shareholders_equity', np.nan)
    debt = combined.get('total_debt', np.nan)
    if pd.notna(equity) and equity > 0 and pd.notna(debt):
        combined['debt_to_equity'] = debt / equity
    
    # Mark provenance
    target_tickers = ','.join(sorted(target_rows.keys()))
    combined['data_provenance'] = 'lookthrough_proforma'
    combined['lookthrough_source'] = target_tickers
    
    return combined


def compute_pro_forma_quarter(
    acquirer_ticker: str,
    as_of_date,
    target_tickers: list[str],
) -> Optional[pd.Series]:
    """
    Compute pro forma combined fundamentals for a single quarter.
    
    Args:
        acquirer_ticker: Acquirer ticker
        as_of_date: Quarter end date
        target_tickers: List of target tickers to combine
    
    Returns:
        Combined row or None if data unavailable
    """
    fund = pd.read_parquet(FUND)
    fund['as_of_date'] = pd.to_datetime(fund['as_of_date']).dt.date
    
    # Get acquirer row
    a_mask = (fund['ticker'] == acquirer_ticker) & (fund['as_of_date'] == as_of_date)
    a_rows = fund[a_mask]
    if a_rows.empty:
        return None
    a_row = a_rows.iloc[0]
    
    # Get target rows
    target_rows = {}
    for t in target_tickers:
        t_mask = (fund['ticker'] == t) & (fund['as_of_date'] == as_of_date)
        t_rows = fund[t_mask]
        if not t_rows.empty:
            target_rows[t] = t_rows.iloc[0]
    
    if not target_rows:
        # No target data available, return standalone
        a_row['data_provenance'] = 'standalone'
        a_row['lookthrough_source'] = None
        return a_row
    
    return combine_quarterly_rows(a_row, target_rows)


# ─────────────────────────────────────────────────────────────────────────────
# PRO FORMA TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────

def get_pro_forma_series(
    acquirer_ticker: str,
    quarters: int = 12,
    include_standalone: bool = True,
) -> pd.DataFrame:
    """
    Get pro forma time series for an acquirer.
    """
    fund = pd.read_parquet(FUND)
    fund['as_of_date'] = pd.to_datetime(fund['as_of_date']).dt.date
    
    # Get acquirer's quarters
    acquirer_data = fund[fund['ticker'] == acquirer_ticker].sort_values('as_of_date')
    if acquirer_data.empty:
        return pd.DataFrame()
    
    # Get acquisitions
    acqs = get_acquisitions_for(acquirer_ticker)
    
    # Pre-process acquisition date ranges
    acq_ranges = []
    if not acqs.empty:
        for _, acq in acqs.iterrows():
            start_raw = acq.get('announcement_date', acq.get('completion_date'))
            end_raw = acq['completion_date']
            
            if pd.isna(start_raw) or pd.isna(end_raw):
                continue
            
            start_date = pd.Timestamp(start_raw).date()
            end_date = pd.Timestamp(end_raw).date() + pd.DateOffset(months=3)
            end_date = pd.Timestamp(end_date).date() if hasattr(end_date, 'date') else end_date
            
            acq_ranges.append({
                'target': acq['target_ticker'],
                'start': start_date,
                'end': end_date,
            })
    
    # For each quarter, determine which targets to combine
    result_rows = []
    
    for _, a_row in acquirer_data.iterrows():
        a_date = a_row['as_of_date']
        if pd.isna(a_date):
            continue
        a_date = pd.Timestamp(a_date).date()
        
        # Find active acquisitions for this date
        active_targets = []
        for acq_range in acq_ranges:
            if acq_range['start'] <= a_date <= acq_range['end']:
                active_targets.append(acq_range['target'])
        
        if active_targets:
            # Combine with active targets
            target_rows = {}
            for t in active_targets:
                t_mask = (fund['ticker'] == t) & (fund['as_of_date'] == a_date)
                t_rows = fund[t_mask]
                if not t_rows.empty:
                    target_rows[t] = t_rows.iloc[0]
                else:
                    # Target data not available, mark as missing
                    target_rows[t] = None
            
            # Filter out None targets (no data available)
            available_targets = {k: v for k, v in target_rows.items() if v is not None}
            
            if available_targets:
                combined = combine_quarterly_rows(a_row, available_targets)
                result_rows.append(combined)
            else:
                # No target data available, use standalone
                if include_standalone:
                    a_row_copy = a_row.copy()
                    a_row_copy['data_provenance'] = 'standalone'
                    a_row_copy['lookthrough_source'] = None
                    result_rows.append(a_row_copy)
        else:
            # No active acquisitions, standalone
            if include_standalone:
                a_row_copy = a_row.copy()
                a_row_copy['data_provenance'] = 'standalone'
                a_row_copy['lookthrough_source'] = None
                result_rows.append(a_row_copy)
    
    if not result_rows:
        return pd.DataFrame()
    
    result = pd.DataFrame(result_rows).sort_values('as_of_date')
    
    # Return last N quarters
    if len(result) > quarters:
        result = result.tail(quarters)
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ACQUISITION RECORD MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def add_acquisition(
    acquirer_ticker: str,
    target_ticker: str,
    completion_date: str,
    announcement_date: str = None,
    purchase_price: float = None,
    consideration_type: str = 'cash',
    notes: str = None,
):
    """
    Add an acquisition record to corporate_actions.
    Enables look-through for the acquirer.
    """
    if CORPORATE_ACTIONS.exists():
        ca = pd.read_parquet(CORPORATE_ACTIONS)
    else:
        ca = pd.DataFrame(columns=[
            'ticker', 'cik', 'entity_name', 'action_type', 'action_date',
            'acquirer_ticker', 'acquirer_cik', 'acquirer_name',
            'target_ticker', 'target_cik', 'target_name',
            'cash_per_share', 'stock_ratio', 'final_price',
            'source', 'filing_date', 'notes',
            'announcement_date', 'completion_date', 'purchase_price',
            'consideration_type', 'lookthrough_start', 'lookthrough_end',
            'pro_forma_method'
        ])
    
    new_record = {
        'ticker': target_ticker,
        'cik': None,
        'entity_name': None,
        'action_type': 'acquisition',
        'action_date': pd.to_datetime(completion_date).date(),
        'acquirer_ticker': acquirer_ticker,
        'acquirer_cik': None,
        'acquirer_name': None,
        'target_ticker': target_ticker,
        'target_cik': None,
        'target_name': None,
        'cash_per_share': None,
        'stock_ratio': None,
        'final_price': None,
        'source': 'manual',
        'filing_date': None,
        'notes': notes,
        'announcement_date': pd.to_datetime(announcement_date).date() if announcement_date else pd.to_datetime(completion_date).date(),
        'completion_date': pd.to_datetime(completion_date).date(),
        'purchase_price': purchase_price,
        'consideration_type': consideration_type,
        'lookthrough_start': pd.to_datetime(announcement_date).date() if announcement_date else pd.to_datetime(completion_date).date(),
        'lookthrough_end': pd.to_datetime(completion_date).date() + pd.DateOffset(months=3),
        'pro_forma_method': 'additive',
    }
    
    ca = pd.concat([ca, pd.DataFrame([new_record])], ignore_index=True)
    ca.to_parquet(CORPORATE_ACTIONS, index=False)
    print(f"Added: {acquirer_ticker} acquires {target_ticker} on {completion_date}")
    print(f"  Look-through: {new_record['lookthrough_start']} → {new_record['lookthrough_end']}")


def detect_acquisitions_from_sec(cik: str, ticker: str) -> list[dict]:
    """
    Detect acquisitions from SEC companyfacts.
    Looks for M&A-related XBRL tags.
    """
    UA = {'User-Agent': 'personal-research derek.moore@example.com'}
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
    
    try:
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200:
            return []
        
        d = r.json()
        facts = d.get('facts', {}).get('us-gaap', {})
        
        acquisitions = []
        
        # Look for business combination tags
        ma_tags = [
            'BusinessCombinationConsiderationTransferred',
            'BusinessAcquisitionPurchasePriceAllocationGoodwillAmount',
            'BusinessCombinationRecognizedIdentifiableAssetsAcquiredGoodwillAndLiabilitiesAssumedNet',
        ]
        
        for tag in ma_tags:
            if tag in facts:
                units = facts[tag].get('units', {}).get('USD', [])
                for entry in units:
                    acquisitions.append({
                        'tag': tag,
                        'date': entry.get('end'),
                        'value': entry.get('val'),
                        'frame': entry.get('frame', ''),
                    })
        
        return acquisitions
    except Exception as e:
        print(f"Error fetching acquisitions for {ticker}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# PRO FORMA FUNDAMENTALS (MAIN ENTRY POINT)
# ─────────────────────────────────────────────────────────────────────────────

def get_pro_forma_fundamentals(
    ticker: str,
    as_of_date: datetime = None,
    include_lookthrough: bool = True,
) -> pd.DataFrame:
    """
    Get fundamentals for a ticker, applying look-through if acquisitions are active.
    
    This is the main entry point for all analytics.
    Use this instead of raw fundamentals.parquet to get pro forma data.
    
    Args:
        ticker: Company ticker
        as_of_date: Specific date (None for latest)
        include_lookthrough: Apply look-through for acquisitions
    
    Returns:
        DataFrame with fundamentals + provenance columns
    """
    fund = pd.read_parquet(FUND)
    fund['as_of_date'] = pd.to_datetime(fund['as_of_date']).dt.date
    
    if not include_lookthrough:
        return fund[fund['ticker'] == ticker].sort_values('as_of_date')
    
    # Get pro forma series
    series = get_pro_forma_series(ticker, quarters=20, include_standalone=True)
    
    if series.empty:
        return fund[fund['ticker'] == ticker].sort_values('as_of_date')
    
    if as_of_date:
        as_of_date = pd.to_datetime(as_of_date).date()
        series = series[series['as_of_date'] <= as_of_date]
    
    return series


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE & TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    acqs = load_acquisitions()
    print(f"corporate_actions acquisitions: {len(acqs)}")
    if acqs.empty:
        print("No acquisition rows. Register via acquisition_backfill.process_acquisition.")
    else:
        fund = pd.read_parquet(FUND) if FUND.exists() else pd.DataFrame()
        have = set(fund["ticker"].astype(str).str.upper()) if not fund.empty else set()
        for _, row in acqs.iterrows():
            acq = str(row.get("acquirer_ticker", "")).upper()
            tgt = str(row.get("target_ticker", "")).upper()
            close = row.get("completion_date")
            tgt_ok = tgt in have
            acq_ok = acq in have
            note = "ok" if tgt_ok and acq_ok else "NO PRO FORMA — missing acquiree/acquirer quarters"
            print(f"  {acq}+{tgt} close={close} acquiree_in_fund={tgt_ok} [{note}]")
            if acq_ok:
                series = get_pro_forma_series(acq, quarters=8)
                n_lt = int((series.get("data_provenance") == "lookthrough_proforma").sum()) if not series.empty and "data_provenance" in series.columns else 0
                print(f"    lookthrough_proforma quarters: {n_lt}")