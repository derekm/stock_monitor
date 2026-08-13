#!/usr/bin/env python3
"""backfill_preferred_fundamentals.py — one-shot additive coverage expander.

What this is
------------
The preferred-metrics history is just a scored view of fundamentals.parquet.
To expand it to max real history and fill the universe you must deepen
fundamentals first, then rebuild the snapshot. This script is that chain,
STRICTLY ADDITIVE (fill NaN + append new (ticker, date); never overwrite
populated cells; EDGAR never displaced).

Order (deepest / best source first so later sources only fill gaps):
  1. backfill_edgar.py            SEC XBRL companyfacts (decades, US filers)
  2. update_fundamentals.py fetch-history
                                  yfinance quarterly (~2y, ADRs / no-CIK)
  3. update_polygon_financials.py Massive/Polygon vX financials (if keyed)
  4. fundamentals_history.py snapshot
                                  rebuild preferred_metrics_history.parquet
  5. preferred_metrics.py --save  latest preferred_metrics.parquet

Usage:
  python backfill_preferred_fundamentals.py
  python backfill_preferred_fundamentals.py --tickers AAPL,GOLD,BTI
  python backfill_preferred_fundamentals.py --no-polygon --no-preferred
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: list[str], timeout: int = 6 * 3600) -> int:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, timeout=timeout)
    if r.returncode != 0:
        print(f"    step exit {r.returncode} (continuing unless fatal)")
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=None,
                    help="Comma-separated subset; default = full universe")
    ap.add_argument("--no-edgar", action="store_true")
    ap.add_argument("--no-yfinance", action="store_true")
    ap.add_argument("--no-polygon", action="store_true")
    ap.add_argument("--no-snapshot", action="store_true")
    ap.add_argument("--no-preferred", action="store_true")
    args = ap.parse_args()

    tflag = ["--tickers", args.tickers] if args.tickers else []
    rc = 0

    if not args.no_edgar:
        rc |= run([PY, str(DATA_DIR / "backfill_edgar.py"), *tflag])
    if not args.no_yfinance:
        cmd = [PY, str(DATA_DIR / "update_fundamentals.py"), "fetch-history"]
        if args.tickers:
            cmd += ["--tickers", args.tickers]
        rc |= run(cmd)
    if not args.no_polygon:
        cmd = [PY, str(DATA_DIR / "update_polygon_financials.py")]
        if args.tickers:
            cmd += ["--tickers", args.tickers]
        rc |= run(cmd)
    if not args.no_snapshot:
        rc |= run([PY, str(DATA_DIR / "fundamentals_history.py"), "snapshot"])
    if not args.no_preferred:
        rc |= run([PY, str(DATA_DIR / "preferred_metrics.py"), "--save"])
    print("\n== preferred-fundamentals backfill finished ==")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
