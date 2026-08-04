#!/usr/bin/env python3
"""
granite_backfill.py — HISTORICAL pre-training of the Granite TTM over the full
daily_prices history for every covered ticker, so the daily `granite_daily.py`
runs start from well-trained models instead of a cold zero-shot base.

This module is now a THIN BACKWARD-COMPATIBLE SHIM over the factored library
`ttm_backfill.py`. All training/windowing logic lives in `ttm_backfill`; this
file re-exports the public API that other scripts import
(`build_full_history_windows`, `train_windows`, `train_aggregate`, `score_windows`,
`per_ticker_plan`, `_clean_price_frame`, `run`, `coverage_report`, `main`, and the
module constants) and delegates `run()` to `ttm_backfill.run_backfill` with the
historical default config.

To set up arbitrary model regimes at global/per-ticker level, use
`ttm_backfill` directly (BackfillConfig / RegimeConfig / Callbacks).

CUDA is used automatically if a CUDA torch build is installed.
"""
from __future__ import annotations

# ---- re-export the library as the public API of this module ----
from ttm_backfill import (  # noqa: F401
    DataConfig, TrainConfig, RegimeConfig, Callbacks, BackfillConfig,
    default_backfill_config, adjusted_backfill_config,
    per_ticker_plan, _clean_price_frame,
    build_windows, train_checkpoint,
    score_windows, run_backfill, compare_adj_unadj,
    CONTEXT, HORIZON, MIN_NEEDED, PRICES, CKPT_DIR, GLOBAL_DIR,
    PADDED_DIR, PER_TICKER_DIR, _device,
)
import ttm_backfill as _lib

# old per-ticker / aggregate trainers both map to the unified train_checkpoint
train_windows = train_checkpoint     # noqa: F401  (per-ticker ckpt)
train_aggregate = train_checkpoint   # noqa: F401  (aggregate ckpt)

import granite_daily as gd          # alias kept for `from granite_backfill import gd`
import window_padding as wp         # alias kept for callers
from pathlib import Path

HERE = Path(__file__).parent
COMPARE_LOG = HERE / "granite_backfill_compare.jsonl"


def build_full_history_windows(prices: "pd.DataFrame", tickers=None,
                               max_windows_per_ticker=200):
    """Backward-compatible wrapper: build per-ticker (context, target) windows
    using the default DataConfig. Delegates to ttm_backfill.build_windows."""
    import pandas as pd
    cfg = DataConfig(max_windows_per_ticker=max_windows_per_ticker,
                     tickers=list(tickers) if tickers else None)
    wbt = build_windows(cfg, prices if isinstance(prices, pd.DataFrame) else pd.read_parquet(prices))
    # return the flat list form the old function returned (for test scripts)
    return [w for wins in wbt.values() for w in wins]


def coverage_report():
    """Report per-ticker window readiness (full 512-context vs short/padded)."""
    import pandas as pd
    df = pd.read_parquet(PRICES)
    plan = per_ticker_plan(df)
    full = [p for p in plan if p[2]]
    padded = [p for p in plan if not p[2]]
    print(f"tickers total: {len(plan)}")
    print(f"  full 512-context (raw history): {len(full)}")
    print(f"  short (<512) -> proxy-padded backfill: {len(padded)}")
    for tk, n, is_full, sector in padded:
        print(f"    PAD {tk:5} n={n:4} sector={sector}")


def run(tickers=None, steps=150, chunk=90, compare=True, batch=None):
    """Historical backfill. Delegates to ttm_backfill.run_backfill with the
    default (unadjusted) config that reproduces the original behavior."""
    cfg = default_backfill_config(
        steps=steps, batch=batch,
        tickers=list(tickers) if tickers else None)
    cfg.compare = compare
    cfg.compare_log = COMPARE_LOG
    return run_backfill(cfg)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    rr = sub.add_parser("run")
    rr.add_argument("--tickers")
    rr.add_argument("--steps", type=int, default=150)
    rr.add_argument("--chunk", type=int, default=90)
    rr.add_argument("--batch", type=int, default=None,
                    help="batch size for GPU steps (default gd.BATCH=8)")
    sub.add_parser("coverage")
    args = ap.parse_args()
    if args.cmd == "coverage":
        coverage_report()
    elif args.cmd == "run":
        run(
            tickers=args.tickers.split(",") if args.tickers else None,
            steps=args.steps,
            chunk=args.chunk,
            batch=args.batch,
        )
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
