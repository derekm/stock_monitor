"""Fetch trailing dividend history (yfinance) for the coverage universe.

Feeds shareholder-yield (item 10): sy = trailing-12m dividends / close.
Writes dividends_cache.parquet (ticker, ex_date, amount). Rate-limited.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent
DELAY_S = 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", default=str(Path.home() / "AppData/Local/Temp/coverage_tickers.txt"))
    ap.add_argument("--out", default="dividends_cache.parquet")
    args = ap.parse_args()

    tickers = [t.strip() for t in Path(args.tickers_file).read_text().splitlines() if t.strip()]
    print(f"{len(tickers)} tickers", flush=True)
    rows = []
    for i, tk in enumerate(tickers, 1):
        try:
            d = yf.Ticker(tk).dividends
            if d is not None and len(d):
                d = d.dropna()
                for dt, amt in d.items():
                    rows.append({"ticker": tk, "ex_date": dt.date(), "amount": float(amt)})
        except Exception as e:  # noqa: BLE001
            print(f"  skip {tk}: {e}", flush=True)
        if i % 25 == 0:
            print(f"{i}/{len(tickers)} ({len(rows)} div rows)", flush=True)
        time.sleep(DELAY_S)
    out = pd.DataFrame(rows)
    out.to_parquet(DATA_DIR / args.out, index=False)
    print(f"Wrote {args.out}: {len(out)} rows, {out['ticker'].nunique()} names with dividends")


if __name__ == "__main__":
    main()
