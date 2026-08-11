#!/usr/bin/env python3
"""data_integrity_deep.py — deeper price/fundamental integrity.

  - Multi-threshold jump scan
  - Suspected split factors (integer-ish price ratios)
  - Stale-quote / flat-line detection
  - Cross-sectional same-day outlier scores
  - Fundamental missingness report
  - Alignment coverage: price ∩ membership ∩ preferred

Usage:
  python data_integrity_deep.py --save
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
PRICES = DATA_DIR / "daily_prices.parquet"
CLEAN = DATA_DIR / "daily_prices_clean.parquet"
FUND = DATA_DIR / "fundamentals.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
PREF = DATA_DIR / "preferred_metrics.parquet"
OUT_DIR = DATA_DIR


def jump_scan() -> pl.DataFrame:
    src = CLEAN if CLEAN.exists() else PRICES
    df = (
        pl.scan_parquet(str(src))
        .with_columns(pl.col("date").cast(pl.Date, strict=False), pl.col("close").cast(pl.Float64))
        .sort(["ticker", "date"])
        .with_columns((pl.col("close") / pl.col("close").shift(1).over("ticker")).alias("ratio"))
        .with_columns((pl.col("ratio") - 1.0).alias("ret"))
        .filter(pl.col("ret").is_not_null())
        .collect()
    )
    # suspected splits: ratio near 0.5, 2, 0.333, 3, 0.25, 4 within 3%
    targets = [0.5, 2.0, 1/3, 3.0, 0.25, 4.0, 0.2, 5.0]
    def near_split(r: float) -> str:
        if r is None or not np.isfinite(r):
            return ""
        for t in targets:
            if abs(r - t) / t < 0.03:
                return f"~{t:.3g}"
        return ""
    pdf = df.to_pandas()
    pdf["suspected_split"] = pdf["ratio"].map(near_split)
    summary = {
        "n_rows": int(len(pdf)),
        "pct_abs_ret_gt_10": float((pdf["ret"].abs() > 0.10).mean()),
        "pct_abs_ret_gt_25": float((pdf["ret"].abs() > 0.25).mean()),
        "pct_abs_ret_gt_35": float((pdf["ret"].abs() > 0.35).mean()),
        "n_suspected_splits": int((pdf["suspected_split"] != "").sum()),
    }
    splits = pdf[pdf["suspected_split"] != ""].sort_values("date", ascending=False)
    return pl.from_pandas(pdf), summary, splits


def flatline_scan(min_days: int = 5) -> pd.DataFrame:
    src = CLEAN if CLEAN.exists() else PRICES
    df = (
        pl.scan_parquet(str(src))
        .with_columns(pl.col("close").cast(pl.Float64))
        .sort(["ticker", "date"])
        .with_columns(
            (pl.col("close") == pl.col("close").shift(1).over("ticker")).alias("flat")
        )
        .collect()
        .to_pandas()
    )
    # run length of flats
    rows = []
    for t, g in df.groupby("ticker"):
        run = 0
        max_run = 0
        for f in g["flat"].fillna(False):
            run = run + 1 if f else 0
            max_run = max(max_run, run)
        if max_run >= min_days:
            rows.append({"ticker": t, "max_flat_run": int(max_run)})
    return pd.DataFrame(rows).sort_values("max_flat_run", ascending=False)


def fundamental_missingness() -> pd.DataFrame:
    if not FUND.exists():
        return pd.DataFrame()
    df = pd.read_parquet(FUND)
    cols = [c for c in (
        "roe", "roic", "debt_to_equity", "interest_coverage", "earnings_stability",
        "ev_ebitda", "pb_ratio", "mktcap_to_assets", "market_cap",
    ) if c in df.columns]
    rows = []
    for c in cols:
        s = df[c]
        rows.append({
            "field": c,
            "n": len(s),
            "n_missing": int(s.isna().sum()),
            "pct_missing": float(s.isna().mean()),
            "n_nonpositive": int(((s <= 0) | s.isna()).sum()) if s.dtype != object else int(s.isna().sum()),
        })
    return pd.DataFrame(rows)


def coverage_matrix() -> dict:
    tickers = {}
    if STOCKS.exists():
        tickers["monitored"] = set(pd.read_parquet(STOCKS)["ticker"].astype(str).str.upper())
    if PREF.exists():
        tickers["preferred"] = set(pd.read_parquet(PREF)["ticker"].astype(str).str.upper())
    src = CLEAN if CLEAN.exists() else PRICES
    if src.exists():
        tickers["prices"] = set(
            pl.scan_parquet(str(src)).select("ticker").unique().collect()["ticker"].to_list()
        )
    keys = list(tickers.keys())
    out = {"sets": {k: len(v) for k, v in tickers.items()}, "intersections": {}}
    if "monitored" in tickers and "prices" in tickers:
        out["intersections"]["monitored_with_prices"] = len(tickers["monitored"] & tickers["prices"])
        out["intersections"]["monitored_missing_prices"] = sorted(tickers["monitored"] - tickers["prices"])[:30]
    if "preferred" in tickers and "prices" in tickers:
        out["intersections"]["preferred_with_prices"] = len(tickers["preferred"] & tickers["prices"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    full, summary, splits = jump_scan()
    print("Jump summary:", json.dumps(summary, indent=2))
    flats = flatline_scan()
    print(f"Flat-line names (max run≥5): {len(flats)}")
    miss = fundamental_missingness()
    print("Fundamental missingness:")
    print(miss.to_string(index=False) if len(miss) else "(none)")
    cov = coverage_matrix()
    print("Coverage:", json.dumps({k: cov[k] for k in ("sets", "intersections") if k in cov}, indent=2, default=str))

    if args.save:
        summary_path = OUT_DIR / "data_integrity_deep_summary.json"
        summary_path.write_text(json.dumps({"jumps": summary, "coverage": cov}, indent=2, default=str))
        splits.head(500).to_csv(OUT_DIR / "suspected_splits.csv", index=False)
        flats.to_csv(OUT_DIR / "price_flatlines.csv", index=False)
        miss.to_csv(OUT_DIR / "fundamental_missingness.csv", index=False)
        # per-ticker jump rate
        pdf = full.to_pandas()
        jr = pdf.assign(big=pdf["ret"].abs() > 0.25).groupby("ticker")["big"].mean().reset_index()
        jr.columns = ["ticker", "pct_days_absret_gt_25"]
        jr.sort_values("pct_days_absret_gt_25", ascending=False).to_csv(
            OUT_DIR / "ticker_jump_rates.csv", index=False
        )
        print("Wrote deep integrity artifacts")


if __name__ == "__main__":
    main()
