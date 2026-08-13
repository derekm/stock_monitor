#!/usr/bin/env python3
"""
add_ticker.py — One-command onboarding of new tickers.

Adds ticker(s) to the universe and runs the full backfill + analytics chain:

  1. Look up name / sector / industry via yfinance (overridable).
  2. Add to monitored_stocks.parquet (idempotent: skips if present).
  3. Backfill price history with --period max (full available history).
  4. Backfill fundamentals via backfill_preferred_fundamentals.py
     (EDGAR XBRL → yfinance quarterly → Polygon financials, additive;
      then rebuild preferred_metrics_history + preferred_metrics).
  5. Compute momentum metrics for the new tickers.
  6. Compute daily market cap (close x shares outstanding).
  7. Run the full analytics pipeline (run_daily_automation.py all jobs).
  8. Regenerate the dashboard export.

Usage:
  python add_ticker.py QSR
  python add_ticker.py QSR CAG PFE
  python add_ticker.py QSR --name "Restaurant Brands" --sector "Consumer Discretionary"
  python add_ticker.py QSR --no-analytics --no-fundamentals   # prices only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
PY = sys.executable


# ── 1. Universe ────────────────────────────────────────────────────────
def load_stocks() -> pd.DataFrame:
    if STOCKS_FILE.exists():
        return pd.read_parquet(STOCKS_FILE)
    return pd.DataFrame(columns=[
        "ticker", "name", "sector", "industry", "subsector",
        "status", "index_member", "notes", "added_date", "last_updated",
    ])


def lookup_meta(ticker: str) -> dict:
    """Fetch name/sector/industry from yfinance; returns {} on failure."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        out = {}
        if info.get("longName"):
            out["name"] = info["longName"]
        elif info.get("shortName"):
            out["name"] = info["shortName"]
        if info.get("sector"):
            out["sector"] = info["sector"]
        if info.get("industry"):
            out["industry"] = info["industry"]
        return out
    except Exception:  # noqa: BLE001
        return {}


def add_to_universe(ticker: str, name: str | None, sector: str | None,
                    industry: str | None, status: str, notes: str) -> bool:
    """Returns True if added, False if already present."""
    df = load_stocks()
    ticker = ticker.upper()
    if ticker in df["ticker"].astype(str).str.upper().values:
        print(f"  [universe] {ticker} already in monitored_stocks — skipping add")
        return False

    meta = lookup_meta(ticker)
    row = {
        "ticker": ticker,
        "name": name or meta.get("name") or ticker,
        "sector": sector or meta.get("sector") or "Unknown",
        "industry": industry or meta.get("industry") or "",
        "subsector": "",
        "status": status,
        "index_member": False,
        "notes": notes or "",
        "added_date": datetime.now().date(),
        "last_updated": pd.Timestamp.now(),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_parquet(STOCKS_FILE, index=False)
    print(f"  [universe] added {ticker}: {row['name']} ({row['sector']})")
    return True


# ── 2-5. Backfill chain ────────────────────────────────────────────────
def run(cmd: list[str], timeout: int = 3600) -> None:
    print(f"\n  >>> {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    tail = (r.stdout or "").strip().splitlines()[-8:]
    for line in tail:
        print(f"      {line}")
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()[-4:]
        for line in err:
            print(f"      !! {line}")
        raise SystemExit(f"step failed ({' '.join(cmd[:2])}): exit {r.returncode}")


def backfill_prices(tickers: list[str]) -> None:
    run([PY, str(DATA_DIR / "backfill_historical.py"),
         "--tickers", ",".join(tickers), "--period", "max"])


def backfill_fundamentals(tickers: list[str]) -> None:
    """Additive EDGAR + yfinance + Polygon, then preferred-metrics snapshot."""
    run([PY, str(DATA_DIR / "backfill_preferred_fundamentals.py"),
         "--tickers", ",".join(tickers)])

def backfill_momentum(tickers: list[str]) -> None:
    """Compute momentum metrics for new tickers immediately."""
    import subprocess, sys
    # Call the build function directly with tickers via a one-liner
    code = f"""
import sys
sys.path.insert(0, r'{DATA_DIR}')
from momentum_analytics import build
df, qdf, ic = build(tickers={tickers})
df.to_parquet(r'{DATA_DIR}/momentum_metrics.parquet', index=False)
qdf.to_parquet(r'{DATA_DIR}/momentum_quintiles.parquet', index=False)
ic.to_parquet(r'{DATA_DIR}/momentum_ic.parquet', index=False)
print(f'Wrote momentum parquets for {len(tickers)} tickers')
"""
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=300)
    tail = (r.stdout or "").strip().splitlines()[-5:]
    for line in tail:
        print(f"      {line}")
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()[-4:]
        for line in err:
            print(f"      !! {line}")
        raise SystemExit(f"momentum backfill failed: exit {r.returncode}")


def marketcap() -> None:
    run([PY, str(DATA_DIR / "add_daily_marketcap.py")])


def run_analytics() -> None:
    run([PY, str(DATA_DIR / "run_daily_automation.py")], timeout=3600 * 6)


def export_dashboard() -> None:
    run([PY, str(DATA_DIR / "export_dashboard_data.py")])


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="+", help="Ticker symbol(s) to onboard")
    ap.add_argument("--name", default=None)
    ap.add_argument("--sector", default=None)
    ap.add_argument("--industry", default=None)
    ap.add_argument("--status", default="monitored",
                    choices=["active", "monitored", "inactive"])
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-fundamentals", action="store_true",
                    help="Skip fundamentals backfill (prices only)")
    ap.add_argument("--no-analytics", action="store_true",
                    help="Skip full analytics + dashboard export")
    args = ap.parse_args()

    tickers = [t.upper() for t in args.tickers]

    print(f"== Onboarding {len(tickers)} ticker(s): {', '.join(tickers)} ==")

    # 1. Universe (single name/sector applies to all when given)
    added = [t for t in tickers if add_to_universe(
        t, args.name, args.sector, args.industry, args.status, args.notes)]
    if not added:
        print("  (all tickers already present)")

    # 2. Prices — max history
    backfill_prices(tickers)

    # 3. Fundamentals
    if not args.no_fundamentals:
        backfill_fundamentals(tickers)

    # 4. Momentum
    if not args.no_fundamentals:
        backfill_momentum(tickers)

    # 5. Market cap
    marketcap()

    # 5. Analytics + dashboard
    if not args.no_analytics:
        run_analytics()
        export_dashboard()

    print(f"\n== Done. {', '.join(tickers)} onboarded with max-history backfill ==")


if __name__ == "__main__":
    main()
