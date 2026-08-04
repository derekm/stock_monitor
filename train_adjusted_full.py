#!/usr/bin/env python3
"""
train_adjusted_full.py — full historical pre-training on ADJUSTED closes.

Now a thin config wrapper over the factored library `ttm_backfill`:
it builds `ttm_backfill.adjusted_backfill_config()` (which mirrors
`default_backfill_config` EXACTLY — same window builder, same 150-step recipe,
same global->padded->per-ticker warm chain — but feeds adj_close windows and
auto-excludes the 95 tickers with no adjustment data) and runs it.

The ONLY variable between this run and the unadjusted `granite_backfill.run()`
is the price source, so the resulting `adjusted_*` checkpoints are directly
comparable to the unadjusted ones in passes 2-4.

Usage:
  python train_adjusted_full.py [--steps 150] [--batch 16]
"""
from __future__ import annotations

import argparse

import ttm_backfill as ttm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150,
                    help="optimizer steps per regime (150 matches the unadjusted run)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--tickers", default=None,
                    help="optional comma list to limit (default: all adj-capable)")
    args = ap.parse_args()

    cfg = ttm.adjusted_backfill_config(steps=args.steps, batch=args.batch)
    if args.tickers:
        cfg.data.tickers = [t.strip().upper() for t in args.tickers.split(",")]
    print(f"adjusted backfill: {len(cfg.data.exclude_tickers)} no-adj tickers excluded; "
          f"{args.steps} steps/regime; out_dir={cfg.data.out_dir}", flush=True)
    ttm.run_backfill(cfg)


if __name__ == "__main__":
    main()
