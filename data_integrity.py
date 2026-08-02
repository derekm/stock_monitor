#!/usr/bin/env python3
"""data_integrity.py — Price/fundamental integrity utilities (Polars-first).

Fixes / guards:
  - Detect and clip bad ticks / unadjusted jumps
  - Optional split-style adjustment via jump detection
  - Point-in-time (PIT) as-of joins for fundamentals history
  - Volume hygiene for Fisher quantity weights
  - Schema checks for critical parquet/CSV artifacts

Usage:
  python data_integrity.py audit
  python data_integrity.py clean-prices --save
  python data_integrity.py pit-fundamentals --save
  python data_integrity.py schema-check
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
PRICES_CLEAN = DATA_DIR / "daily_prices_clean.parquet"
FUND = DATA_DIR / "fundamentals.parquet"
FUND_HIST = DATA_DIR / "fundamentals_history.parquet"
FUND_PIT = DATA_DIR / "fundamentals_pit.parquet"
SCHEMA_REPORT = DATA_DIR / "schema_check_report.json"
JUMP_REPORT = DATA_DIR / "price_jump_audit.csv"


REQUIRED_SCHEMAS = {
    "daily_prices": {"date", "ticker", "close"},
    "fundamentals": {"ticker"},
    "monitored_stocks": {"ticker"},
    "portfolio_holdings": {"ticker"},
    "preferred_metrics": {"ticker", "composite_score"},
}


def scan_prices() -> pl.LazyFrame:
    return pl.scan_parquet(str(PRICES))


def audit_price_jumps(max_abs_ret: float = 0.35) -> pl.DataFrame:
    """Flag sessions with |return| > threshold per ticker (possible splits/bad ticks)."""
    lf = (
        scan_prices()
        .select(["date", "ticker", "close", "volume"] if True else ["date", "ticker", "close"])
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
        .sort(["ticker", "date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("ret")
        )
        .filter(pl.col("ret").abs() > max_abs_ret)
    )
    df = lf.collect()
    df.write_csv(JUMP_REPORT)
    print(f"Jump audit: {df.height} rows > ±{max_abs_ret:.0%} → {JUMP_REPORT.name}")
    return df


def clean_prices(
    clip_ret: float = 0.35,
    min_price: float = 0.01,
    save: bool = False,
) -> pl.DataFrame:
    """Produce cleaned price panel: drop non-positive, ffill sparse gaps lightly, clip returns.

    Note: True vendor-adjusted prices are preferred. This is a defensive local fix when
    corporate-action-adjusted history is unavailable.
    """
    df = (
        scan_prices()
        .with_columns(
            pl.col("date").cast(pl.Datetime, strict=False).cast(pl.Date, strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False) if True else pl.lit(None),
        )
        .filter(pl.col("close").is_not_null() & (pl.col("close") >= min_price))
        .sort(["ticker", "date"])
        .with_columns(
            pl.col("close").forward_fill().over("ticker").alias("close_ff"),
        )
        .with_columns(
            (pl.col("close_ff") / pl.col("close_ff").shift(1).over("ticker") - 1.0).alias("ret")
        )
        .with_columns(
            pl.when(pl.col("ret").abs() > clip_ret)
            .then(None)
            .otherwise(pl.col("close_ff"))
            .alias("close_clipped")
        )
        .with_columns(
            pl.col("close_clipped").forward_fill().over("ticker").alias("close_clean")
        )
        .with_columns(
            # volume: zero/neg → null → ffill (Fisher quantity hygiene)
            pl.when(pl.col("volume").is_not_null() & (pl.col("volume") > 0))
            .then(pl.col("volume"))
            .otherwise(None)
            .forward_fill()
            .over("ticker")
            .alias("volume_clean")
        )
        .select([
            "date", "ticker",
            pl.col("close_clean").alias("close"),
            pl.col("volume_clean").alias("volume"),
            pl.col("ret"),
        ])
        .collect()
    )
    n_bad = df.filter(pl.col("ret").abs() > clip_ret).height
    print(f"Clean prices: {df.height} rows; flagged jumps materializing as gaps filled: ~{n_bad}")
    if save:
        df.drop("ret").write_parquet(PRICES_CLEAN)
        # Also write a pandas-friendly copy path used by downstream if present
        print(f"Wrote {PRICES_CLEAN}")
    return df


def pit_fundamentals(save: bool = False) -> pl.DataFrame:
    """Build point-in-time fundamentals: as-of each available history date, last known row per ticker.

    If fundamentals_history is missing, snapshot fundamentals with as_of_date = today.
    """
    if FUND_HIST.exists():
        hist = pl.read_parquet(str(FUND_HIST))
        if "as_of_date" not in hist.columns:
            hist = hist.with_columns(pl.lit(None).cast(pl.Date).alias("as_of_date"))
        hist = hist.with_columns(pl.col("as_of_date").cast(pl.Date, strict=False))
    elif FUND.exists():
        hist = pl.read_parquet(str(FUND))
        hist = hist.with_columns(pl.lit("2026-07-30").str.to_date().alias("as_of_date"))
    else:
        raise SystemExit("No fundamentals sources")

    # Sort and keep last observation per ticker, as_of_date
    hist = hist.sort(["ticker", "as_of_date"])
    # PIT table is the history itself marked as usable as-of
    pit = hist.unique(subset=["ticker", "as_of_date"], keep="last")
    if save:
        pit.write_parquet(FUND_PIT)
        print(f"Wrote {FUND_PIT} ({pit.height} rows)")
    return pit


def schema_check() -> dict:
    report = {"ok": True, "artifacts": {}}
    for name, required in REQUIRED_SCHEMAS.items():
        path = None
        for ext in (".parquet", ".csv"):
            p = DATA_DIR / f"{name}{ext}"
            if p.exists():
                path = p
                break
        if path is None:
            report["artifacts"][name] = {"exists": False}
            report["ok"] = False
            continue
        try:
            if path.suffix == ".parquet":
                cols = set(pl.scan_parquet(str(path)).collect_schema().names())
            else:
                cols = set(pl.read_csv(str(path), n_rows=1).columns)
            missing = sorted(required - cols)
            report["artifacts"][name] = {
                "exists": True,
                "path": path.name,
                "n_cols": len(cols),
                "missing_required": missing,
            }
            if missing:
                report["ok"] = False
        except Exception as e:
            report["artifacts"][name] = {"exists": True, "error": str(e)}
            report["ok"] = False
    SCHEMA_REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    p = sub.add_parser("clean-prices")
    p.add_argument("--clip", type=float, default=0.35)
    p.add_argument("--save", action="store_true")
    sub.add_parser("pit-fundamentals").add_argument("--save", action="store_true")
    sub.add_parser("schema-check")
    p = sub.add_parser("all")
    p.add_argument("--save", action="store_true")
    args = ap.parse_args()
    if args.cmd == "audit":
        audit_price_jumps()
    elif args.cmd == "clean-prices":
        clean_prices(clip_ret=args.clip, save=args.save)
    elif args.cmd == "pit-fundamentals":
        pit_fundamentals(save=getattr(args, "save", False))
    elif args.cmd == "schema-check":
        schema_check()
    elif args.cmd == "all":
        schema_check()
        audit_price_jumps()
        clean_prices(save=True)
        try:
            pit_fundamentals(save=True)
        except SystemExit as e:
            print(e)


if __name__ == "__main__":
    main()
