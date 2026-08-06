#!/usr/bin/env python3
"""
fundamentals_history.py — Time-series fundamentals & preferred-metric snapshots.

Problem: inclusion screens need history, not only the latest row.
Solution:
  1. fundamentals.parquet already stores dated rows (as_of_date) — append-only updates
  2. This module:
       - backfills synthetic history for robust thesis backtests
       - snapshots preferred_metrics scores through time
       - evaluates screen pass/fail on each as_of_date for backtesting

Usage:
  python fundamentals_history.py backfill --quarters 8
  python fundamentals_history.py snapshot
  python fundamentals_history.py backtest-screens
  python fundamentals_history.py show --ticker MOS
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
SNAP = DATA_DIR / "preferred_metrics_history.parquet"
SNAP_CSV = DATA_DIR / "preferred_metrics_history.csv"
SCREEN_BT = DATA_DIR / "screen_backtest.csv"

# thresholds (same as preferred_metrics)
from analytics_common import (
    ROE_MIN, ROIC_MIN, DE_MAX, EV_MAX, PB_MAX, MCA_MAX,
    quality_value_parts, COMP_W_Q, COMP_W_V,
)


def load_fund() -> pd.DataFrame:
    df = pd.read_parquet(FUND)
    # `as_of_date` is DATE on disk -> read as datetime.date; keep it a date.
    return df


def save_fund(df: pd.DataFrame) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), FUND)


def latest(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)


def backfill_quarters(n_quarters: int = 8, seed: int = 42) -> None:
    """
    Create prior quarter-end snapshots by mean-reverting noise around latest values.
    Enables screen backtests when only one real snapshot exists.
    Mark source with fundamentals_history_backfill.
    """
    df = load_fund()
    lat = latest(df)
    rng = np.random.default_rng(seed)
    # quarter ends going back
    last = lat["as_of_date"].max()
    # quarter-end dates as datetime.date (ingest as date, not Timestamp)
    dates = [d.date() for d in pd.date_range(end=last, periods=n_quarters + 1, freq="QE")]
    # drop the last if equals latest
    dates = [d for d in dates if d < last + timedelta(days=1)]

    metric_cols = [
        "roe", "roic", "debt_to_equity", "interest_coverage", "earnings_stability",
        "pb_ratio", "ev_ebitda", "mktcap_to_assets", "market_cap_b", "total_assets_b",
    ]
    rows = []
    for d in dates:
        for _, r in lat.iterrows():
            nr = r.copy()
            nr["as_of_date"] = d
            nr["source"] = "fundamentals_history_backfill"
            nr["quality_source"] = "backfill_from_latest"
            nr["last_updated"] = pd.Timestamp.now()
            # mild mean-reverting noise
            for c in metric_cols:
                if c in nr and pd.notna(nr[c]):
                    noise = rng.normal(0, 0.08)
                    # ratios stay positive-ish
                    val = float(nr[c]) * (1 + noise)
                    if c in ("roe", "roic", "earnings_stability"):
                        val = float(np.clip(val, -0.5, 2.0))
                    elif c in ("pb_ratio", "ev_ebitda", "mktcap_to_assets", "debt_to_equity"):
                        val = float(max(val, 0.01))
                    nr[c] = val
            if pd.notna(nr.get("market_cap_b")) and pd.notna(nr.get("total_assets_b")):
                nr["market_cap"] = float(nr["market_cap_b"]) * 1e9
                nr["total_assets"] = float(nr["total_assets_b"]) * 1e9
            rows.append(nr)

    # remove prior backfill for same dates/tickers then append
    hist = df[df.get("source") != "fundamentals_history_backfill"] if "source" in df.columns else df
    # also drop any existing rows on those quarter dates to avoid dups
    hist = hist[~hist["as_of_date"].isin(dates)]
    out = pd.concat([hist, pd.DataFrame(rows)], ignore_index=True)
    save_fund(out)
    print(f"Backfilled {len(dates)} quarter-ends × {len(lat)} tickers → {FUND}")
    print(f"  dates: {[d.date() for d in dates]}")
    print(f"  total rows now: {len(out)}")


def score_row(r: pd.Series) -> dict:
    roe, roic, de = r.get("roe"), r.get("roic"), r.get("debt_to_equity")
    ev, pb, mca = r.get("ev_ebitda"), r.get("pb_ratio"), r.get("mktcap_to_assets")
    buffett = (
        pd.notna(roe) and roe >= ROE_MIN
        and pd.notna(roic) and roic >= ROIC_MIN
        and pd.notna(de) and de <= DE_MAX
    )
    trifecta = (
        pd.notna(ev) and ev <= EV_MAX
        and pd.notna(pb) and pb <= PB_MAX
        and pd.notna(mca) and mca <= MCA_MAX
    )
    # simplified scores aligned with preferred_metrics
    # composite via the canonical weighted formula (weights in analytics_common)
    q, v = quality_value_parts(
        roe=roe, roic=roic, de=de, earnings_stability=r.get("earnings_stability"),
        ev=ev, pb=pb, mca=mca,
    )
    composite = COMP_W_Q * q + COMP_W_V * v
    if buffett and trifecta:
        composite = min(1.0, composite + 0.08)
    decision = (
        "INCLUDE_CORE" if buffett and trifecta else
        "INCLUDE_VALUE" if trifecta else
        "INCLUDE_QUALITY" if buffett else
        "SATELLITE" if composite >= 0.50 else
        "WATCH" if composite >= 0.35 else
        "AVOID"
    )
    return {
        "buffett_pass": buffett,
        "trifecta_pass": trifecta,
        "quality_score": round(q, 4),
        "value_score": round(v, 4),
        "composite_score": round(composite, 4),
        "decision": decision,
    }


def snapshot_all_dates() -> pd.DataFrame:
    df = load_fund()
    rows = []
    for _, r in df.iterrows():
        s = score_row(r)
        rows.append({
            "as_of_date": r["as_of_date"],
            "ticker": r["ticker"],
            "roe": r.get("roe"),
            "roic": r.get("roic"),
            "debt_to_equity": r.get("debt_to_equity"),
            "ev_ebitda": r.get("ev_ebitda"),
            "pb_ratio": r.get("pb_ratio"),
            "mktcap_to_assets": r.get("mktcap_to_assets"),
            "source": r.get("source"),
            **s,
        })
    out = pd.DataFrame(rows).sort_values(["as_of_date", "ticker"])
    out.to_csv(SNAP_CSV, index=False)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), SNAP)
    print(f"Snapshot history → {SNAP} ({len(out)} rows, {out.as_of_date.nunique()} dates)")
    return out


def backtest_screens() -> pd.DataFrame:
    snap = snapshot_all_dates()
    # coverage through time
    g = snap.groupby("as_of_date").agg(
        n=("ticker", "count"),
        buffett=("buffett_pass", "sum"),
        trifecta=("trifecta_pass", "sum"),
        dual=("decision", lambda s: (s == "INCLUDE_CORE").sum()),
        value=("decision", lambda s: (s == "INCLUDE_VALUE").sum()),
        quality=("decision", lambda s: (s == "INCLUDE_QUALITY").sum()),
        median_composite=("composite_score", "median"),
    ).reset_index()
    g.to_csv(SCREEN_BT, index=False)
    print("\n=== Screen membership through time ===")
    print(g.to_string(index=False))
    print(f"\nWrote {SCREEN_BT}")
    return g


def show_ticker(ticker: str) -> None:
    df = load_fund()
    t = df[df.ticker == ticker.upper()].sort_values("as_of_date")
    if t.empty:
        print("No rows for", ticker)
        return
    cols = [c for c in ["as_of_date", "roe", "roic", "debt_to_equity", "pb_ratio", "ev_ebitda",
                        "mktcap_to_assets", "source"] if c in t.columns]
    print(t[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("backfill")
    p.add_argument("--quarters", type=int, default=8)
    sub.add_parser("snapshot")
    sub.add_parser("backtest-screens")
    p = sub.add_parser("show")
    p.add_argument("--ticker", required=True)
    args = ap.parse_args()
    if args.cmd == "backfill":
        backfill_quarters(args.quarters)
    elif args.cmd == "snapshot":
        snapshot_all_dates()
    elif args.cmd == "backtest-screens":
        backtest_screens()
    elif args.cmd == "show":
        show_ticker(args.ticker)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
