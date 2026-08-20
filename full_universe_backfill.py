#!/usr/bin/env python3
"""
full_universe_backfill.py — Run unified EDGAR extraction across all monitored tickers.

Process:
1. Load CIK map and monitored stocks
2. For each ticker, extract financials using XBRL + HTML fallback
3. Merge into fundamentals.parquet (additive: never overwrite existing data)
4. Track provenance for every value
5. Checkpoint/resume capability via backfill_checkpoints

Usage:
  python full_universe_backfill.py
  python full_universe_backfill.py --max-tickers 100
  python full_universe_backfill.py --dry-run
  python full_universe_backfill.py --no-resume  # force fresh start
"""

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

from edgar_lib import load_cik_map, get_cik, CIK_OVERRIDES, NO_COMPANYFACTS
from unified_edgar_pipeline import unified_extract
from backfill_checkpoints import CheckpointManager


DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
MONITORED = DATA_DIR / "monitored_stocks.parquet"
BACKUP_DIR = DATA_DIR / "backfill_backups"
BACKUP_DIR.mkdir(exist_ok=True)


def load_fundamentals() -> pd.DataFrame:
    """Load existing fundamentals or return empty DataFrame."""
    if FUND.exists():
        df = pd.read_parquet(FUND)
        if "as_of_date" in df.columns:
            df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
        return df
    return pd.DataFrame()


def save_fundamentals(df: pd.DataFrame):
    """Save fundamentals to parquet."""
    df = df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    df = df.sort_values(["ticker", "as_of_date"]).drop_duplicates(
        subset=["ticker", "as_of_date"], keep="first"
    )
    atomic_write_parquet(df, FUND)
    print(f"Saved {len(df)} rows to {FUND}")


def backup_fundamentals():
    """Create timestamped backup."""
    if FUND.exists():
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        backup = BACKUP_DIR / f"fundamentals_backup_{ts}.parquet"
        import shutil
        shutil.copy2(FUND, backup)
        print(f"Backup saved: {backup}")
        return backup
    return None


def merge_additive(existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    """
    Merge new rows into existing DataFrame additively.
    
    Rules:
    - Never overwrite existing (ticker, date) rows with same source priority
    - Fill NaN cells in existing rows with new values
    - For FCF, prefer computed over proxy/unavailable
    - Append brand new (ticker, date) rows
    """
    if not new_rows:
        return existing
    
    new_df = pd.DataFrame(new_rows)
    
    if existing.empty:
        return new_df
    
    # Normalize dates
    existing["as_of_date"] = pd.to_datetime(existing["as_of_date"]).dt.date
    new_df["as_of_date"] = pd.to_datetime(new_df["as_of_date"]).dt.date
    
    # Set index for merge
    existing_idx = existing.set_index(["ticker", "as_of_date"])
    new_idx = new_df.set_index(["ticker", "as_of_date"])
    
    # Find overlapping indices
    overlap = existing_idx.index.intersection(new_idx.index)
    
    if len(overlap):
        # For overlapping rows, fill NaN cells
        for col in new_idx.columns:
            if col not in existing_idx.columns:
                existing_idx[col] = None
            
            missing = existing_idx.loc[overlap, col].isna()
            if missing.any():
                take = new_idx.loc[overlap, col].where(missing)
                existing_idx.loc[overlap, col] = existing_idx.loc[overlap, col].fillna(take)
        
        # FCF provenance upgrade: prefer computed > proxy > unavailable
        if "fcf_provenance" in existing_idx.columns and "fcf_provenance" in new_idx.columns:
            for idx in overlap:
                old_prov = existing_idx.loc[idx, "fcf_provenance"]
                new_prov = new_idx.loc[idx, "fcf_provenance"]
                
                if new_prov == "computed" and (pd.isna(old_prov) or old_prov in ("proxy", "unavailable")):
                    # Upgrade FCF and related fields
                    for upgrade_col in ["free_cash_flow", "fcf_provenance", "capital_expenditure_ttm",
                                       "capital_expenditure_ttm", "fcf_source"]:
                        if upgrade_col in new_idx.columns:
                            val = new_idx.loc[idx, upgrade_col]
                            if pd.notna(val):
                                existing_idx.loc[idx, upgrade_col] = val
    
    # Append new rows
    brand_new = new_idx[~new_idx.index.isin(existing_idx.index)]
    if len(brand_new):
        combined = pd.concat([existing_idx, brand_new], ignore_index=False)
    else:
        combined = existing_idx
    
    return combined.reset_index()


def run_full_backfill(max_tickers: int = None, dry_run: bool = False, resume: bool = True):
    """Run backfill across all monitored tickers with checkpoint support."""
    
    # Load CIK map
    cik_map = load_cik_map()
    cik_map.update(CIK_OVERRIDES)
    
    # Load tickers from fundamentals (full universe)
    existing_fund = load_fundamentals()
    if not existing_fund.empty:
        tickers = sorted(existing_fund["ticker"].astype(str).str.upper().unique().tolist())
    elif MONITORED.exists():
        stocks = pd.read_parquet(MONITORED)
        tickers = sorted(stocks["ticker"].astype(str).str.upper().unique().tolist())
    else:
        tickers = sorted(cik_map.keys())
    
    tickers = [t for t in tickers if t not in NO_COMPANYFACTS]
    
    if max_tickers:
        tickers = tickers[:max_tickers]
    
    print(f"Full Universe Backfill Plan:")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Estimated time: ~{len(tickers) * 0.5:.0f} seconds ({len(tickers) * 0.5 / 60:.1f} minutes)")
    print(f"  Backup: Yes")
    print(f"  Incremental save: Every 50 tickers")
    print(f"  Resume: {resume}")
    
    if dry_run:
        print(f"\nDRY RUN - would process {len(tickers)} tickers")
        for t in tickers[:5]:
            cik = get_cik(t, cik_map)
            print(f"  {t}: CIK={cik}")
        return
    
    # Initialize checkpoint manager
    source_paths = [FUND, MONITORED]
    cm = CheckpointManager("full_universe_backfill", BACKUP_DIR.parent / "backfill_checkpoints")
    session_info = cm.start_session(tickers, source_paths=source_paths, force_restart=not resume)
    
    # Backup
    backup_fundamentals()
    
    # Load existing fundamentals
    existing = load_fundamentals()
    print(f"  Existing rows: {len(existing)}")
    
    # Get pending tickers from checkpoint
    pending_tickers = cm.get_pending_tickers()
    print(f"  Pending tickers: {len(pending_tickers)} (completed: {len(cm.get_completed_tickers())})")
    
    # Process pending tickers
    all_new_rows = []
    stats = {"ok": 0, "error": 0, "no_cik": 0, "no_data": 0}
    
    start_time = time.time()
    
    for i, ticker in enumerate(pending_tickers, 1):
        # Find original index for progress display
        orig_idx = tickers.index(ticker) + 1
        cik = get_cik(ticker, cik_map)
        if cik is None:
            stats["no_cik"] += 1
            cm.update_ticker(ticker, last_date=None, rows_processed=0, status="failed")
            continue
        
        try:
            result = unified_extract(cik, ticker, use_html=True, max_html_quarters=2)
            
            if result["merged_count"] > 0:
                # Compute market_cap from price × shares
                if 'shares_outstanding' in result and result['shares_outstanding'] > 0:
                    # Get latest price for this ticker
                    from analytics_common import load_adj_prices_pandas, atomic_write_parquet
                    prices = load_adj_prices_pandas(tickers=[ticker])
                    if len(prices) > 0:
                        latest_price = prices.iloc[-1].get('close', None)
                        if latest_price and latest_price > 0:
                            result['market_cap'] = latest_price * result['shares_outstanding']
                
                all_new_rows.extend(result["rows"])
                stats["ok"] += 1
                
                # Update checkpoint with last processed date
                if result["rows"]:
                    last_date = max(r["as_of_date"] for r in result["rows"])
                    cm.update_ticker(ticker, last_date=last_date, rows_processed=len(result["rows"]), status="completed")
                
                if i % 10 == 0:
                    print(f"  [{orig_idx}/{len(tickers)}] {ticker}: {result['merged_count']} rows, "
                          f"quality={result['quality_score']}")
            else:
                stats["no_data"] += 1
                cm.update_ticker(ticker, last_date=None, rows_processed=0, status="failed")
                print(f"  [{orig_idx}/{len(tickers)}] {ticker}: no data")
            
        except Exception as e:
            stats["error"] += 1
            cm.update_ticker(ticker, last_date=None, rows_processed=0, status="failed")
            print(f"  [{orig_idx}/{len(tickers)}] {ticker}: ERROR - {e}")
        
        # Incremental save every 50 tickers
        if i % 50 == 0 and all_new_rows:
            print(f"\n--- Incremental save at ticker {i} ---")
            existing = merge_additive(existing, all_new_rows)
            save_fundamentals(existing)
            all_new_rows = []
            print(f"  Stats so far: {stats}")
            print(f"  Time elapsed: {time.time() - start_time:.0f}s\n")
        
        time.sleep(0.12)
    
    # Final merge and save
    if all_new_rows:
        existing = merge_additive(existing, all_new_rows)
    
    save_fundamentals(existing)
    
    elapsed = time.time() - start_time
    
    # Complete session
    summary = cm.complete_session()
    
    print(f"\n{'='*60}")
    print(f"BACKFILL COMPLETE")
    print(f"{'='*60}")
    print(f"  Tickers processed: {len(tickers)}")
    print(f"  OK: {stats['ok']}")
    print(f"  No CIK: {stats['no_cik']}")
    print(f"  No data: {stats['no_data']}")
    print(f"  Errors: {stats['error']}")
    print(f"  Total rows: {len(existing)}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    # Save stats
    with open("backfill_stats.json", "w") as f:
        json.dump({
            "tickers_processed": len(tickers),
            "stats": stats,
            "total_rows": len(existing),
            "elapsed_seconds": elapsed,
            "timestamp": pd.Timestamp.now().isoformat(),
            "checkpoint_summary": summary,
        }, f, indent=2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tickers", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="Force fresh start, ignore checkpoint")
    args = ap.parse_args()
    
    run_full_backfill(max_tickers=args.max_tickers, dry_run=args.dry_run, resume=not args.no_resume)