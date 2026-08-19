#!/usr/bin/env python3
"""
scan_ifrs_filers.py — find universe tickers whose facts live in ifrs-full.

Foreign private issuers file 40-F or 20-F under IFRS. Their companyfacts carries an
ifrs-full block, and any us-gaap block is absent or a stale remnant -- Barrick (B)
has 248 us-gaap tags ending 2010-12-31 beside 301 ifrs-full tags current to
2025-12-31. Reading us-gaap alone yields no revenue and a decade-stale balance sheet.

Reports, per ticker: which taxonomies exist, the newest fact date in each, and which
one the extractor selects. Use it to size the population before and after a change to
taxonomy handling.

Usage:
    python scan_ifrs_filers.py --limit 300
    python scan_ifrs_filers.py --tickers B,STM,CHKP,ABBNY
    python scan_ifrs_filers.py --limit 500 --json ifrs_filers.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent
PRICES = DATA_DIR / "daily_prices.parquet"


def newest_end(block: dict) -> str:
    newest = ""
    for tag in block.values():
        for arr in tag.get("units", {}).values():
            for e in arr:
                end = e.get("end", "")
                if end > newest:
                    newest = end
    return newest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--json")
    args = ap.parse_args()

    import edgar_companyfacts_v2 as V

    cik_map = V.load_cik_map()
    cik_map.update(V.CIK_OVERRIDES)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        u = (pl.read_parquet(PRICES, columns=["ticker"])["ticker"]
             .unique().sort().to_list())
        tickers = [t for t in u if t in cik_map][: args.limit]

    print(f"scanning {len(tickers)} tickers for taxonomy\n")
    rows = []
    ifrs_only = []
    for t in tickers:
        cik = cik_map.get(t)
        if not cik:
            continue
        try:
            d = V.fetch_companyfacts(cik)
        except Exception:
            continue
        if not d:
            continue
        facts = d.get("facts", {})
        g = facts.get("us-gaap") or {}
        i = facts.get("ifrs-full") or {}
        if not i:
            continue                      # us-gaap only: nothing to report
        ge, ie = newest_end(g), newest_end(i)
        picked = "ifrs-full" if ie > ge else "us-gaap"
        rows.append({"ticker": t, "us_gaap_tags": len(g), "us_gaap_newest": ge,
                     "ifrs_tags": len(i), "ifrs_newest": ie, "picked": picked})
        flag = ""
        if picked == "ifrs-full" and g:
            # the case that silently produced stale data when only us-gaap was read
            flag = "  <-- us-gaap block is STALE"
            ifrs_only.append(t)
        elif picked == "ifrs-full":
            flag = "  <-- ifrs only"
            ifrs_only.append(t)
        print(f"  {t:8} us-gaap {len(g):4} tags to {ge or '-':10}   "
              f"ifrs {len(i):4} tags to {ie or '-':10}  -> {picked}{flag}")

    print()
    print(f"tickers with an ifrs-full block : {len(rows)}")
    print(f"tickers where ifrs-full is newer: {len(ifrs_only)}")
    if ifrs_only:
        print(f"  {','.join(ifrs_only)}")

    if args.json and rows:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
