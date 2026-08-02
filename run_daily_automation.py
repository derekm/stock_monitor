#!/usr/bin/env python3
"""
run_daily_automation.py — Master daily job for stock_monitor analytics stack.

Runs (in order):
  1. preferred_metrics
  2. inclusion_criteria (+ defensive exploration)
  3. stress_dual_pass
  4. rolling_window_analysis
  5. allpairs_correlations
  6. fundamentals_history snapshot + screen backtest
  7. dupont_analysis
  8. growth_tech_analytics (optional light)
  9. export_dashboard_data

Usage:
  python run_daily_automation.py
  python run_daily_automation.py --skip-growth --skip-allpairs
  python run_daily_automation.py --only export,inclusion
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent

JOBS = [
    ("preferred", ["preferred_metrics.py", "--save"]),
    ("inclusion", ["inclusion_criteria.py", "--explore-defensive", "--save"]),
    ("stress", ["stress_dual_pass.py", "--save"]),
    ("crisis", ["crisis_correlation.py", "--save"]),
    ("factor_rot", ["factor_rotation_defense.py", "--save"]),
    ("risk_enrich", ["risk_enrich.py"]),
    ("rolling", ["rolling_window_analysis.py", "--universe", "portfolio", "--save"]),
    ("rolling_corr", ["rolling_correlation_windows.py", "--save"]),
    ("tail_hedge", ["tail_risk_hedging.py", "--save"]),
    ("allpairs", ["allpairs_correlations.py", "--window", "63", "--step", "21", "--max-assets", "50"]),
    ("fund_snap", ["fundamentals_history.py", "snapshot"]),
    ("screen_bt", ["fundamentals_history.py", "backtest-screens"]),
    ("dupont", ["dupont_analysis.py", "--save"]),
    ("growth", ["growth_tech_analytics.py"]),
    ("export", ["export_dashboard_data.py"]),
]


def run_job(name: str, args: list[str], timeout: int = 300) -> bool:
    print(f"\n══ {name} ══")
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, *args],
            cwd=str(DATA_DIR),
            timeout=timeout,
        )
        dt = time.time() - t0
        ok = r.returncode == 0
        print(f"{'OK' if ok else 'FAIL'} {name} in {dt:.1f}s (code={r.returncode})")
        return ok
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT {name}")
        return False
    except Exception as e:
        print(f"ERROR {name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-growth", action="store_true")
    ap.add_argument("--skip-allpairs", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of job names")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    results = {}
    for name, cmd in JOBS:
        if only and name not in only:
            continue
        if args.skip_growth and name == "growth":
            continue
        if args.skip_allpairs and name == "allpairs":
            continue
        results[name] = run_job(name, cmd)

    print("\n══ SUMMARY ══")
    for k, v in results.items():
        print(f"  {'✓' if v else '✗'} {k}")
    failed = [k for k, v in results.items() if not v]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
