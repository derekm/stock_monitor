#!/usr/bin/env python3
"""
run_daily_automation.py — Master daily job for stock_monitor analytics stack.

Runs (in order):
  1. hmm_regime_detection      (regime states → feeds rebalance, constraints, hedges)
  2. rebalance_calendar        (month-end schedule + regime-driven turnover band)
  3. preferred_metrics         (dual-pass screen → preferred_metrics.csv)
  4. inclusion_criteria        (defensive/quality/value exploration)
  5. stress_dual_pass          (scenario analysis on dual-pass)
  6. crisis_correlation        (calm vs crisis corr breakdown)
  7. factor_rotation_defense   (factor sleeve performance)
  8. risk_enrich               (enrich fundamentals)
  9. rolling_window_analysis   (portfolio 63d metrics + screen stability)
  10. rolling_correlation_windows (rolling corr + corr stability)
  11. tail_risk_hedging        (regime-aware hedge overlays)
  12. allpairs_correlations    (asset/sector pairwise corr history)
  13. fundamentals_history     (snapshot + screen backtest)
  14. dupont_analysis          (ROE decomposition)
  15. growth_tech_analytics    (growth tech sleeve corr/perf)
  16. peer_analytics           (cross-stock peer comparison + signals)
  17. earnings_catalyst        (pre-earnings momentum + drift + IV-vs-realized)
  18. pair_engine              (cointegration pair trades, walk-forward OOS)
  19. cross_section            (multi-factor sector-neutral L/S)
  20. aggregate                (signal_aggregator: OOS IC-weighted composite)
  21. export_dashboard_data    (dashboard_data/data.json)

Usage:
  python run_daily_automation.py
  python run_daily_automation.py --skip-growth --skip-allpairs
  python run_daily_automation.py --only hmm,rebalance,preferred,export
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent

JOBS = [
    ("hmm", ["hmm_regime_detection.py", "--n-states", "3", "--save"]),
    ("rebalance", ["rebalance_calendar.py", "--months", "18", "--save"]),
    ("preferred", ["preferred_metrics.py", "--save"]),
    ("inclusion", ["inclusion_criteria.py", "--explore-defensive", "--save"]),
    ("stress", ["stress_dual_pass.py", "--save"]),
    ("crisis", ["crisis_correlation.py", "--save"], 600),
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
    ("peer", ["peer_analytics.py", "--save"]),
    ("earnings", ["earnings_catalyst.py", "--save"]),
    ("pairs", ["pair_engine.py", "--save"], 900),
    ("cross", ["cross_section.py", "--save"], 600),
    ("aggregate", ["signal_aggregator.py", "--save"]),
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
        print(f"TIMEOUT {name} (limit={timeout}s)")
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
    for job in JOBS:
        name = job[0]
        cmd = job[1]
        timeout = job[2] if len(job) > 2 else 300
        if only and name not in only:
            continue
        if args.skip_growth and name == "growth":
            continue
        if args.skip_allpairs and name == "allpairs":
            continue
        results[name] = run_job(name, cmd, timeout)

    print("\n══ SUMMARY ══")
    for k, v in results.items():
        print(f"  {'✓' if v else '✗'} {k}")
    failed = [k for k, v in results.items() if not v]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
