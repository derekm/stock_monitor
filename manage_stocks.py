#!/usr/bin/env python3
"""
manage_stocks.py - Maintain the monitored_stocks.parquet master table.

Usage:
  python manage_stocks.py list
  python manage_stocks.py add --ticker TICK --name "Name" --sector Materials --industry "..." --subsector "..." --status active --index_member
  python manage_stocks.py set_status TICK active|monitored|inactive
  python manage_stocks.py set_index TICK true|false
"""

import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"

def load_stocks():
    if STOCKS_FILE.exists():
        return pd.read_parquet(STOCKS_FILE)
    return pd.DataFrame(columns=[
        "ticker", "name", "sector", "industry", "subsector",
        "status", "index_member", "notes", "added_date", "last_updated"
    ])

def save_stocks(df):
    df["last_updated"] = pd.Timestamp.now()
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, STOCKS_FILE)
    print(f"Saved {len(df)} stocks to {STOCKS_FILE}")

def cmd_list(args):
    df = load_stocks()
    if args.status:
        df = df[df["status"] == args.status]
    if args.sector:
        df = df[df["sector"] == args.sector]
    print(df.to_string(index=False))
    print(f"\nTotal: {len(df)}")

def cmd_add(args):
    df = load_stocks()
    if args.ticker.upper() in df["ticker"].values:
        print(f"{args.ticker} already exists. Use set_status or edit manually.")
        return
    new = {
        "ticker": args.ticker.upper(),
        "name": args.name,
        "sector": args.sector,
        "industry": args.industry or "",
        "subsector": args.subsector or "",
        "status": args.status,
        "index_member": args.index_member,
        "notes": args.notes or "",
        "added_date": datetime.now().date(),
        "last_updated": pd.Timestamp.now(),
    }
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    save_stocks(df)
    print(f"Added {args.ticker.upper()}")
    
    if args.backfill:
        run_full_backfill(args.ticker.upper(), args.days)


def run_full_backfill(ticker: str, days: int = 4000):
    """Run complete backfill pipeline for a new ticker."""
    import subprocess
    import sys
    
    print(f"\n{'='*60}")
    print(f"Running full backfill for {ticker}...")
    print(f"{'='*60}")
    
    # Step 1: Fetch prices
    print(f"\n[1/4] Fetching {days} days of price data...")
    result = subprocess.run([
        sys.executable, "update_prices.py", "fetch", "--days", str(days)
    ], cwd=DATA_DIR, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f"Warning: price fetch returned {result.returncode}")
    
    # Step 2: Fetch fundamentals (current snapshot)
    print(f"\n[2/4] Fetching current fundamentals...")
    result = subprocess.run([
        sys.executable, "update_fundamentals.py", "fetch", "--tickers", ticker
    ], cwd=DATA_DIR, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Step 3: Fetch EDGAR history (deep fundamentals)
    print(f"\n[3/5] Fetching EDGAR history...")
    result = subprocess.run([
        sys.executable, "update_fundamentals.py", "fetch-history", "--tickers", ticker
    ], cwd=DATA_DIR, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Step 4: Compute daily market cap from shares (so market_cap is populated
    # for the new ticker immediately — otherwise it's NaN until the next full
    # automation run). Depends on steps 1-3 (prices + shares in fundamentals).
    print(f"\n[4/5] Computing daily market cap from shares...")
    result = subprocess.run([
        sys.executable, "add_daily_marketcap.py"
    ], cwd=DATA_DIR, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Step 5: Run daily automation (preferred, peer, implied_r, momentum, inclusion, aggregate, technical)
    print(f"\n[5/5] Running analytics pipeline...")
    result = subprocess.run([
        sys.executable, "run_daily_automation.py", 
        "--only", "preferred,peer,implied_r,momentum,inclusion,aggregate,technical"
    ], cwd=DATA_DIR, capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    print(f"\n{'='*60}")
    print(f"Full backfill complete for {ticker}")
    print(f"{'='*60}")

def cmd_set_status(args):
    df = load_stocks()
    mask = df["ticker"] == args.ticker.upper()
    if not mask.any():
        print(f"Ticker {args.ticker} not found")
        return
    df.loc[mask, "status"] = args.status
    save_stocks(df)
    print(f"Set {args.ticker.upper()} status to {args.status}")

def cmd_set_index(args):
    df = load_stocks()
    mask = df["ticker"] == args.ticker.upper()
    if not mask.any():
        print(f"Ticker {args.ticker} not found")
        return
    df.loc[mask, "index_member"] = args.value.lower() in ("true", "1", "yes")
    save_stocks(df)
    print(f"Set {args.ticker.upper()} index_member to {df.loc[mask, 'index_member'].iloc[0]}")

def main():
    parser = argparse.ArgumentParser(description="Manage monitored stocks table")
    sub = parser.add_subparsers(dest="cmd")

    p_app = sub.add_parser("apply-json", help="Apply Manage-tab staged JSON")
    p_app.add_argument("--file", required=True)
    p_app.set_defaults(func=cmd_apply_json)

    p_list = sub.add_parser("list")
    p_list.add_argument("--status", choices=["active", "monitored", "inactive"])
    p_list.add_argument("--sector")
    p_list.set_defaults(func=cmd_list)

    p_add = sub.add_parser("add")
    p_add.add_argument("--ticker", required=True)
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--sector", required=True)
    p_add.add_argument("--industry", default="")
    p_add.add_argument("--subsector", default="")
    p_add.add_argument("--status", default="monitored", choices=["active", "monitored", "inactive"])
    p_add.add_argument("--index_member", action="store_true")
    p_add.add_argument("--notes", default="")
    p_add.add_argument("--backfill", action="store_true", help="Run full backfill pipeline after adding (prices, fundamentals, EDGAR, analytics)")
    p_add.add_argument("--days", type=int, default=4000, help="Days of price history to fetch (default 4000)")
    p_add.set_defaults(func=cmd_add)

    p_status = sub.add_parser("set_status")
    p_status.add_argument("ticker")
    p_status.add_argument("status", choices=["active", "monitored", "inactive"])
    p_status.set_defaults(func=cmd_set_status)

    p_idx = sub.add_parser("set_index")
    p_idx.add_argument("ticker")
    p_idx.add_argument("value")
    p_idx.set_defaults(func=cmd_set_index)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


def cmd_apply_json(args):
    """Apply staged updates from Manage tab JSON export."""
    import json
    path = Path(args.file)
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    df = load_stocks()
    for row in data:
        t = str(row.get("ticker", "")).upper().strip()
        if not t:
            continue
        mask = df["ticker"].astype(str).str.upper() == t
        if mask.any():
            for k in ("index_member", "defensive_value_index", "growth_tech_index", "in_portfolio"):
                if k in row and k in df.columns:
                    df.loc[mask, k] = bool(row[k])
            for k in ("growth_sleeve", "value_sleeve", "sector", "industry", "name", "notes", "status"):
                if k in row and row[k] is not None and k in df.columns:
                    df.loc[mask, k] = row[k]
        else:
            new = {c: None for c in df.columns}
            new["ticker"] = t
            for k, v in row.items():
                if k in new:
                    new[k] = v
            if "added_date" in df.columns:
                new["added_date"] = datetime.now().strftime("%Y-%m-%d")
            if "status" in new and not new.get("status"):
                new["status"] = "monitored"
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        print(f"Applied {t}")
    save_stocks(df)


if __name__ == "__main__":
    main()

