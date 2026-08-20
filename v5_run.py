#!/usr/bin/env python3
"""
v5_run.py — run one honest expanding-window V5 pass on the real feature store.

"Honest" means the numbers reported are out-of-sample by construction:
  - the ranker trains on an expanding window and predicts the NEXT block only
  - an embargo of `embargo_dates` sits between train and test, so a label built
    from forward returns cannot overlap the training window
  - conformal calibration is fit on past data and applied forward
  - IC is computed on the out-of-sample scores against realized forward returns
  - book Sharpe comes from the backtest of those OOS weights, after costs

Nothing here tunes on the test blocks, so the reported IC and Sharpe are what the
configuration actually produced -- not a best case selected after the fact.

Usage:
    python v5_run.py
    python v5_run.py --min-train-dates 504 --test-block 21 --horizons 1,5,21
    python v5_run.py --gross 1.0 --max-name 0.05 --json v5_result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-root", default="./v5_feature_store")
    ap.add_argument("--min-train-dates", type=int, default=504)
    ap.add_argument("--test-block", type=int, default=21)
    ap.add_argument("--step", type=int, default=21)
    ap.add_argument("--embargo", type=int, default=5)
    ap.add_argument("--horizons", default="1,5,21")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gross", type=float, default=1.0)
    ap.add_argument("--max-name", type=float, default=0.05)
    ap.add_argument("--max-short", type=float, default=0.03)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--json")
    args = ap.parse_args()

    from v5_integrated import V5Config, V5Pipeline

    horizons = tuple(int(h) for h in args.horizons.split(","))
    cfg = V5Config(
        lgb_num_threads=10,
        store_root=args.store_root,
        horizons=horizons,
        min_train_dates=args.min_train_dates,
        test_block=args.test_block,
        step=args.step,
        embargo_dates=args.embargo,
        conformal_alpha=args.alpha,
        gross_target=args.gross,
        max_name_weight=args.max_name,
        max_short_weight=args.max_short,
        lgb_num_boost_round=args.rounds,
        run_name="v5_real_honest",
        save_artifacts=True,
    )

    print("CONFIG (the numbers below are produced by exactly this)")
    print(f"  horizons           {horizons}")
    print(f"  min_train_dates    {cfg.min_train_dates}")
    print(f"  test_block / step  {cfg.test_block} / {cfg.step}")
    print(f"  embargo_dates      {cfg.embargo_dates}")
    print(f"  conformal_alpha    {cfg.conformal_alpha}")
    print(f"  gross / name cap   {cfg.gross_target} / {cfg.max_name_weight}")
    print()

    pipe = V5Pipeline(cfg)
    summary = pipe.run()

    print()
    print("=" * 62)
    print("MEASURED RESULT")
    print("=" * 62)
    print(json.dumps(summary, indent=1, default=str)[:4000])

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
