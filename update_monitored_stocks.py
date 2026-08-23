#!/usr/bin/env python3
"""
update_monitored_stocks.py — Sync monitored_stocks with daily_prices universe,
fetching sector/industry from yfinance for new tickers.

Usage:
  python update_monitored_stocks.py              # dry-run (show what would be added)
  python update_monitored_stocks.py --save       # commit changes
  python update_monitored_stocks.py --all        # process ALL tickers in daily_prices (not just missing)
  python update_monitored_stocks.py --force      # update sector/industry on EXISTING tickers
  python update_monitored_stocks.py --all --force --save  # full sync with overwrite
  python update_monitored_stocks.py --max-new 50 --save   # limit new additions per run
"""
from __future__ import annotations
import argparse
from datetime import date, datetime
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent
MONITORED = DATA_DIR / "monitored_stocks.parquet"
PRICES = DATA_DIR / "daily_prices.parquet"

# Defaults for new rows
DEFAULT_STATUS = "active"
DEFAULT_INDEX_MEMBER = False
DEFAULT_IN_PORTFOLIO = False
DEFAULT_DEFENSIVE_VALUE = False
DEFAULT_GROWTH_TECH = False
DEFAULT_GROWTH_SLEEVE = None
DEFAULT_VALUE_SLEEVE = None
DEFAULT_DUAL_PASS = False
DEFAULT_INSTRUMENT_TYPE = "stock"
DEFAULT_SPY500_MEMBER = False
DEFAULT_SPY500_SECTOR = None
DEFAULT_SPY500_DATE_ADDED = None


def classify_instrument_type(ticker: str) -> str:
    """NASDAQ/NYSE ticker structure. Warrants/units/preferreds are not common stock."""
    t = str(ticker).upper().strip()
    if not t:
        return "stock"
    if t.endswith("-WT") or t.endswith("-WS") or t.endswith("-W"):
        return "warrant"
    if t.endswith("-U") or t.endswith("-UN") or t.endswith("-UU"):
        return "unit"
    if t.endswith("-R"):
        return "right"
    if "-" in t:
        suf = t.split("-", 1)[1]
        if suf in {"WT", "WTS", "WS", "W"}:
            return "warrant"
        if suf in {"U", "UN", "UU"}:
            return "unit"
        if suf and suf.isalpha() and len(suf) <= 2:
            return "preferred"
        return "stock"
    if len(t) >= 5:
        if t.endswith(("WW", "WS", "WT")):
            return "warrant"
        last = t[-1]
        if last == "W":
            return "warrant"
        if last == "U":
            return "unit"
        if last == "R":
            return "right"
        if last == "F":
            return "otc_foreign"
        if last == "Y":
            return "adr"
    return "stock"


def fetch_yfinance_info(ticker: str, max_retries: int = 3) -> dict | None:
    """Fetch sector/industry from yfinance with retries."""
    for attempt in range(max_retries):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            sector = info.get("sector")
            industry = info.get("industry")
            name = info.get("longName") or info.get("shortName") or ticker
            quote_type = info.get("quoteType")
            if sector or industry or name != ticker:
                return {"name": name, "sector": sector, "industry": industry, "quoteType": quote_type}
            return None
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  !! {ticker}: failed after {max_retries} attempts ({e})")
                return None
            time.sleep(1.5 * (attempt + 1))  # exponential backoff
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="Write changes to parquet")
    ap.add_argument("--all", action="store_true", help="Process ALL tickers in daily_prices (not just missing)")
    ap.add_argument("--force", action="store_true", help="Update sector/industry on EXISTING monitored tickers")
    ap.add_argument("--max-new", type=int, default=None, help="Max new tickers to add this run (ignored with --all)")
    ap.add_argument("--delay", type=float, default=0.5, help="Delay between yfinance calls (seconds)")
    ap.add_argument("--reclass-instruments", action="store_true",
                    help="Recompute instrument_type from ticker structure (no yfinance)")
    args = ap.parse_args()

    # Load existing monitored stocks
    if MONITORED.exists():
        monitored = pd.read_parquet(MONITORED)
        existing_tickers = set(monitored["ticker"].astype(str).str.upper().unique())
        print(f"Existing monitored tickers: {len(existing_tickers)}")
    else:
        monitored = pd.DataFrame(columns=[
            "ticker", "name", "sector", "industry", "subsector",
            "status", "index_member", "notes", "added_date", "last_updated",
            "in_portfolio", "defensive_value_index", "growth_tech_index",
            "growth_sleeve", "value_sleeve", "dual_pass_member",
            "instrument_type", "sp500_member", "sp500_sector", "sp500_date_added"
        ])
        existing_tickers = set()
        print("No existing monitored_stocks.parquet - starting fresh")

    if args.reclass_instruments:
        if monitored.empty:
            print("No monitored_stocks to reclass.")
            return
        monitored["instrument_type"] = monitored["ticker"].map(classify_instrument_type)
        print(monitored["instrument_type"].value_counts().to_string())
        if args.save:
            monitored.to_parquet(MONITORED, index=False)
            print(f"Wrote {len(monitored)} rows → {MONITORED}")
        else:
            print("Dry-run. Use --save to write.")
        return

    # Get all tickers from daily_prices
    print("Reading daily_prices universe...")
    prices = pd.read_parquet(PRICES, columns=["ticker"])
    universe_tickers = sorted(prices["ticker"].astype(str).str.upper().unique())
    print(f"Total tickers in daily_prices: {len(universe_tickers)}")

    # Determine which tickers to process
    if args.all:
        tickers_to_process = universe_tickers
        print(f"--all specified: processing ALL {len(tickers_to_process)} tickers")
    else:
        missing = [t for t in universe_tickers if t not in existing_tickers]
        if args.max_new:
            missing = missing[:args.max_new]
        tickers_to_process = missing
        print(f"Missing from monitored_stocks: {len(missing)}")
        if not missing:
            print("All tickers already monitored. Nothing to do.")
            return

    # Fetch sector/industry for tickers
    new_rows = []
    updated_rows = []
    today = date.today()
    now_ts = pd.Timestamp.now()

    for i, ticker in enumerate(tickers_to_process, 1):
        is_existing = ticker in existing_tickers
        action = "Updating" if (is_existing and args.force) else ("Adding" if not is_existing else "Skipping")
        
        if is_existing and not args.force:
            print(f"[{i}/{len(tickers_to_process)}] {ticker}: already monitored (use --force to update)")
            continue

        print(f"[{i}/{len(tickers_to_process)}] {action} {ticker}...", end=" ", flush=True)
        info = fetch_yfinance_info(ticker)

        if info:
            sector = info.get("sector")
            industry = info.get("industry")
            # Use yfinance sector/industry directly (no custom mapping)
            name = info.get("name", ticker)
        else:
            sector = None
            industry = None
            name = ticker

        row = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "industry": industry,
            "subsector": industry,  # use industry as subsector
            "status": DEFAULT_STATUS,
            "index_member": DEFAULT_INDEX_MEMBER,
            "notes": f"Auto-synced from daily_prices via yfinance on {today}",
            "added_date": today,
            "last_updated": now_ts,
            "in_portfolio": DEFAULT_IN_PORTFOLIO,
            "defensive_value_index": DEFAULT_DEFENSIVE_VALUE,
            "growth_tech_index": DEFAULT_GROWTH_TECH,
            "growth_sleeve": DEFAULT_GROWTH_SLEEVE,
            "value_sleeve": DEFAULT_VALUE_SLEEVE,
            "dual_pass_member": DEFAULT_DUAL_PASS,
            "instrument_type": classify_instrument_type(ticker),
            "sp500_member": DEFAULT_SPY500_MEMBER,
            "sp500_sector": DEFAULT_SPY500_SECTOR,
            "sp500_date_added": DEFAULT_SPY500_DATE_ADDED,
        }

        if is_existing:
            updated_rows.append(row)
            print(f"OK ({sector} / {industry})")
        else:
            new_rows.append(row)
            print(f"OK ({sector} / {industry})")

        if args.delay > 0 and i < len(tickers_to_process):
            time.sleep(args.delay)

    # Summary
    print(f"\nNew tickers to add: {len(new_rows)}")
    print(f"Existing tickers to update: {len(updated_rows)}")
    total_with_sector = sum(1 for r in new_rows + updated_rows if r["sector"])
    total_without_sector = len(new_rows) + len(updated_rows) - total_with_sector
    print(f"  With sector: {total_with_sector}")
    print(f"  Without sector: {total_without_sector}")

    if not new_rows and not updated_rows:
        print("No changes to make.")
        return

    if args.save:
        if updated_rows:
            # Update existing rows
            updated_df = pd.DataFrame(updated_rows)
            for _, row in updated_df.iterrows():
                mask = monitored["ticker"] == row["ticker"]
                for col in updated_df.columns:
                    if col not in ["ticker", "added_date"]:  # don't overwrite added_date
                        monitored.loc[mask, col] = row[col]
            print(f"Updated {len(updated_rows)} existing tickers")

        if new_rows:
            # Add new rows
            new_df = pd.DataFrame(new_rows)
            # Ensure column order matches existing
            new_df = new_df[monitored.columns.tolist()] if not monitored.empty else new_df
            monitored = pd.concat([monitored, new_df], ignore_index=True)
            print(f"Added {len(new_rows)} new tickers")

        # Sort and deduplicate
        monitored = monitored.drop_duplicates(subset=["ticker"], keep="first")
        monitored = monitored.sort_values("ticker").reset_index(drop=True)
        monitored.to_parquet(MONITORED, index=False)
        print(f"\n✓ Written {len(monitored)} total tickers to {MONITORED}")
    else:
        print("\nDry-run complete. Use --save to commit changes.")


if __name__ == "__main__":
    main()