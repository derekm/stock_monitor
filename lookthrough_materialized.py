#!/usr/bin/env python3
"""
lookthrough_materialized.py — Materialized Pro Forma View for Analytics.

Creates a pre-computed materialized view of pro-forma fundamentals for all tickers
with active acquisitions. This view is used by peer_analytics, signal_aggregator,
and other modules that need pro-forma peer comparisons.

Output: lookthrough_materialized.parquet
  - One row per ticker per quarter
  - Columns: ticker, as_of_date, plus all fundamental columns
  - Provenance columns: data_provenance, lookthrough_source
  - Includes both standalone and pro-forma rows
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
CORPORATE_ACTIONS = DATA_DIR / "corporate_actions.parquet"
OUT = DATA_DIR / "lookthrough_materialized.parquet"


def build_materialized_view() -> pd.DataFrame:
    """Build materialized pro-forma view for all tickers with acquisitions."""
    print("Building lookthrough materialized view...")
    
    # Load fundamentals
    fund = pd.read_parquet(FUND)
    fund["as_of_date"] = pd.to_datetime(fund["as_of_date"]).dt.date
    
    # Load acquisitions
    if not CORPORATE_ACTIONS.exists():
        print("  No corporate_actions.parquet, returning standalone fundamentals")
        fund["data_provenance"] = "standalone"
        fund["lookthrough_source"] = None
        return fund
    
    ca = pd.read_parquet(CORPORATE_ACTIONS)
    acqs = ca[ca["action_type"].isin(["acquisition", "merger"])].copy()
    
    if acqs.empty:
        print("  No acquisitions found, returning standalone fundamentals")
        fund["data_provenance"] = "standalone"
        fund["lookthrough_source"] = None
        return fund
    
    print(f"  Found {len(acqs)} acquisition records")
    
    # Build result
    result_rows = []
    
    # Process each acquirer
    for acquirer in acqs["acquirer_ticker"].unique():
        acquirer_data = fund[fund["ticker"] == acquirer].copy()
        if acquirer_data.empty:
            continue
        
        acquirer_acqs = acqs[acqs["acquirer_ticker"] == acquirer].sort_values("completion_date")
        
        # Process each quarter for this acquirer
        for _, a_row in acquirer_data.iterrows():
            a_date = a_row["as_of_date"]
            if pd.isna(a_date):
                continue
            
            # Find active acquisitions for this date
            active_targets = []
            for _, acq in acquirer_acqs.iterrows():
                start_raw = acq.get("announcement_date", acq.get("completion_date"))
                end_raw = acq["completion_date"]
                
                if pd.isna(start_raw) or pd.isna(end_raw):
                    continue
                
                start_date = pd.Timestamp(start_raw).date()
                end_date = pd.Timestamp(end_raw).date() + pd.DateOffset(months=3)
                end_date = pd.Timestamp(end_date).date() if hasattr(end_date, "date") else end_date
                
                if start_date <= a_date <= end_date:
                    active_targets.append(acq["target_ticker"])
            
            if active_targets:
                # Combine with active targets
                target_rows = {}
                for t in active_targets:
                    t_mask = (fund["ticker"] == t) & (fund["as_of_date"] == a_date)
                    t_rows = fund[t_mask]
                    if not t_rows.empty:
                        target_rows[t] = t_rows.iloc[0]
                
                available_targets = {k: v for k, v in target_rows.items() if v is not None}
                
                if available_targets:
                    # Create pro-forma row
                    combined = a_row.copy()
                    combined["data_provenance"] = "lookthrough_proforma"
                    combined["lookthrough_source"] = ",".join(sorted(available_targets.keys()))
                    
                    # Additive combination for income statement / balance sheet items
                    additive_cols = [
                        "total_revenue", "operating_income", "net_income",
                        "free_cash_flow", "total_assets", "total_debt",
                        "shareholders_equity", "cash_and_equivalents",
                        "total_liabilities", "capital_expenditure",
                        "ebitda", "gross_profit", "interest_expense",
                    ]
                    
                    for col in additive_cols:
                        if col in combined.index and pd.notna(combined[col]):
                            base_val = pd.to_numeric(combined[col], errors="coerce")
                            if pd.notna(base_val):
                                for t_row in available_targets.values():
                                    t_val = pd.to_numeric(t_row.get(col), errors="coerce")
                                    if pd.notna(t_val):
                                        combined[col] = base_val + t_val
                    
                    # Per-share metrics: use weighted average by shares
                    shares_cols = ["shares_outstanding", "common_shares"]
                    for sc in shares_cols:
                        if sc in combined.index:
                            base_shares = pd.to_numeric(combined[sc], errors="coerce")
                            for t_row in available_targets.values():
                                t_shares = pd.to_numeric(t_row.get(sc), errors="coerce")
                                if pd.notna(base_shares) and pd.notna(t_shares):
                                    combined[sc] = base_shares + t_shares
                                    break
                    
                    result_rows.append(combined.to_dict())
                else:
                    # No target data available, use standalone
                    a_row_copy = a_row.copy()
                    a_row_copy["data_provenance"] = "standalone"
                    a_row_copy["lookthrough_source"] = None
                    result_rows.append(a_row_copy.to_dict())
            else:
                # No active acquisitions, standalone
                a_row_copy = a_row.copy()
                a_row_copy["data_provenance"] = "standalone"
                a_row_copy["lookthrough_source"] = None
                result_rows.append(a_row_copy.to_dict())
    
    # Add all other tickers (no acquisitions) as standalone
    acquirer_tickers = set(acqs["acquirer_ticker"].unique())
    other_tickers = fund[~fund["ticker"].isin(acquirer_tickers)].copy()
    other_tickers["data_provenance"] = "standalone"
    other_tickers["lookthrough_source"] = None
    result_rows.extend(other_tickers.to_dict("records"))
    
    if not result_rows:
        return pd.DataFrame()
    
    result = pd.DataFrame(result_rows).sort_values(["ticker", "as_of_date"]).reset_index(drop=True)
    
    print(f"  Materialized view: {len(result)} rows, {result['data_provenance'].value_counts().to_dict()}")
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build lookthrough materialized view")
    ap.add_argument("--save", action="store_true", help="Save to parquet")
    ap.add_argument("--show", action="store_true", help="Show summary")
    args = ap.parse_args()
    
    df = build_materialized_view()
    
    if args.show or not args.save:
        print(f"\nTotal rows: {len(df)}")
        print(f"Tickers: {df['ticker'].nunique()}")
        print(f"Date range: {df['as_of_date'].min()} to {df['as_of_date'].max()}")
        print(f"Provenance: {df['data_provenance'].value_counts().to_dict()}")
        if "lookthrough_source" in df.columns:
            lt = df[df["data_provenance"] == "lookthrough_proforma"]
            print(f"Lookthrough rows: {len(lt)}")
            if len(lt) > 0:
                print(f"  Sources: {lt['lookthrough_source'].value_counts().head(10).to_dict()}")
    
    if args.save:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUT)
        print(f"\nSaved {len(df)} rows → {OUT}")


if __name__ == "__main__":
    main()