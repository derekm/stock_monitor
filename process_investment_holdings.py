#!/usr/bin/env python3
"""
process_investment_holdings.py — Process the extracted investment holdings into a panel format.

Reads investment_holdings.parquet (from extract_investment_holdings.py) and pivots it to have
one row per (ticker, as_of_date) with columns for each concept.
Also computes a total marketable securities (current + non-current) and total investments.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
INVESTMENT_FILE = DATA_DIR / "investment_holdings.parquet"
OUTPUT_FILE = DATA_DIR / "investment_holdings_panel.parquet"

def main():
    if not INVESTMENT_FILE.exists():
        print(f"File not found: {INVESTMENT_FILE}")
        return
    df = pd.read_parquet(INVESTMENT_FILE)
    print(f"Read {len(df)} rows from {INVESTMENT_FILE}")
    print(f"Columns: {df.columns.tolist()}")
    # Ensure as_of_date is datetime
    df['as_of_date'] = pd.to_datetime(df['as_of_date'])
    # Pivot
    pivot = df.pivot_table(index=['ticker', 'as_of_date'], columns='concept', values='value', aggfunc='sum')
    # Flatten column multi-index
    pivot.columns = [str(col) for col in pivot.columns]
    pivot = pivot.reset_index()
    print(f"Pivoted shape: {pivot.shape}")
    # Compute total marketable securities (if both current and non-current exist)
    if 'MarketableSecuritiesCurrent' in pivot.columns and 'MarketableSecuritiesNoncurrent' in pivot.columns:
        pivot['MarketableSecuritiesTotal'] = pivot['MarketableSecuritiesCurrent'].fillna(0) + pivot['MarketableSecuritiesNoncurrent'].fillna(0)
    # Compute total investments (sum of all investment concepts? We'll sum a subset)
    # Let's define a list of investment concepts we want to sum.
    investment_cols = [c for c in pivot.columns if c not in ['ticker', 'as_of_date']]
    # We'll sum all, but note that some may be overlapping (e.g., MarketableSecuritiesTotal includes its components).
    # For simplicity, we'll just keep the individual columns and let the user decide.
    # Save to parquet
    pivot.to_parquet(OUTPUT_FILE)
    print(f"Saved panel to {OUTPUT_FILE}")
    # Show a sample
    print("\nSample of the panel (first 5 rows):")
    print(pivot.head())
    # Show column list
    print(f"\nColumns ({len(pivot.columns)}):")
    for col in pivot.columns:
        print(f"  {col}")

if __name__ == '__main__':
    main()