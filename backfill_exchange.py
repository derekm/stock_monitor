#!/usr/bin/env python3
"""backfill_exchange.py — batched exchange backfill for monitored_stocks.parquet
Resume-safe: saves after each batch. Run with --delay 0.3
"""
import argparse, time
from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent
MONITORED = DATA_DIR / "monitored_stocks.parquet"

def fetch_exchange(ticker: str, max_retries=3):
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info
            ex = info.get("exchange") or info.get("fullExchangeName")
            # normalize fullExchangeName variants to short codes
            if ex and len(ex) > 5 and ex not in {"NasdaqGS","NasdaqGM","NasdaqCM"}:
                # yfinance sometimes returns "NasdaqGS - NasdaqGS" etc — keep first token
                ex = ex.split()[0]
            return ex
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  !! {ticker}: {e}")
                return None
            time.sleep(1.5*(attempt+1))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=500, help="save after N tickers")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--max", type=int, default=None, help="max tickers this run (for pilot)")
    ap.add_argument("--save", action="store_true", help="write parquet")
    args = ap.parse_args()

    df = pd.read_parquet(MONITORED)
    if "exchange" not in df.columns:
        df["exchange"] = pd.NA
    # resume: only those still NA
    mask = df["exchange"].isna()
    todo = df.loc[mask, "ticker"].astype(str).str.upper().tolist()
    print(f"Total monitored {len(df)}, missing exchange {len(todo)}")
    if args.max:
        todo = todo[:args.max]
        print(f"Pilot max {args.max} -> {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return
    updated = 0
    for i, ticker in enumerate(todo, 1):
        ex = fetch_exchange(ticker)
        df.loc[df["ticker"].astype(str).str.upper() == ticker, "exchange"] = ex
        updated += 1
        print(f"[{i}/{len(todo)}] {ticker} -> {ex}")
        if args.delay and i < len(todo):
            time.sleep(args.delay)
        if i % args.batch == 0:
            if args.save:
                df.to_parquet(MONITORED, index=False)
                print(f"  -> checkpoint {i}/{len(todo)} saved to {MONITORED}")
            else:
                print(f"  -> checkpoint {i}/{len(todo)} (dry-run)")
    if args.save:
        df.to_parquet(MONITORED, index=False)
        print(f"Done. Updated {updated} tickers -> {MONITORED}")
        vc = df["exchange"].value_counts(dropna=False).head(20)
        print(vc.to_string())
    else:
        print(f"Dry-run done. Would have updated {updated}. Use --save to commit.")

if __name__ == "__main__":
    main()
