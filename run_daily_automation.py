#!/usr/bin/env python3
"""
run_daily_automation.py — Master daily job for stock_monitor analytics stack.

Runs 21 jobs as a dependency DAG with multiprocessing (independent jobs run
in parallel; only real dependencies serialize). Falls back to sequential
execution if multiprocessing is unavailable.

Dependency order (edges = must-finish-before):
  hmm → rebalance
  preferred → inclusion → stress → allpairs
  preferred → risk_enrich → rolling → rolling_corr → tail_hedge
  preferred → dupont → growth
  preferred → peer
  growth/peer → earnings → pairs → cross → aggregate → technical → export
  econ_cal / est_rev are independent; shadow runs after preferred+aggregate

Usage:
  python run_daily_automation.py
  python run_daily_automation.py --only hmm,rebalance,preferred,export
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent

# name -> (cmd, timeout_s)
JOBS = {
    "hmm": (["hmm_regime_detection.py", "--n-states", "3", "--save"], None),
    "market_cap": (["add_daily_marketcap.py"], None),
    "rebalance": (["rebalance_calendar.py", "--months", "18", "--save"], None),
    "preferred": (["preferred_metrics.py", "--save"], None),
    "implied_r": (["implied_r_screen.py", "--save"], None),
    "momentum": (["momentum_analytics.py", "--save"], None),
    "inclusion": (["inclusion_criteria.py", "--explore-defensive", "--save"], None),
    "stress": (["stress_dual_pass.py", "--save"], None),
    "crisis": (["crisis_correlation.py", "--save"], None),
    "factor_rot": (["factor_rotation_defense.py", "run", "--save"], None),
    "risk_enrich": (["risk_enrich.py"], None),
    "rolling": (["rolling_window_analysis.py", "--universe", "all", "--save"], None),
    "rolling_corr": (["rolling_correlation_windows.py", "--save"], None),
    "tail_hedge": (["tail_risk_hedging.py", "--save"], None),
    "allpairs": (["allpairs_correlations.py", "--window", "63", "--step", "21", "--max-assets", "50"], None),
    "fund_snap": (["fundamentals_history.py", "snapshot"], None),
    "screen_bt": (["fundamentals_history.py", "backtest-screens"], None),
    "edgar_backfill": (["full_universe_backfill.py", "--max-tickers", "200", "--resume"], None),
    "backfill_new_tickers": (["acquisition_backfill.py", "backfill_new_tickers_job"], None),
    "dupont": (["dupont_analysis.py", "--save"], None),
    "growth": (["growth_tech_analytics.py"], None),
    "peer": (["peer_analytics.py", "--save"], None),
    "earnings": (["earnings_catalyst.py", "--save"], None),
    "pairs": (["pair_engine.py", "--save"], None),
    "cross": (["cross_section.py", "--save"], None),
    "aggregate": (["signal_aggregator.py", "--save"], None),
    "technical": (["technical_signals.py", "--save"], None),
    "econ_cal": (["economic_calendar.py", "--save"], None),
    "est_rev": (["estimate_revisions.py", "--save"], None),
    "shadow": (["shadow_book.py", "--save"], None),
    "damodaran": (["damodaran_quality.py", "--all"], None),
    "lookthrough": (["lookthrough_engine.py"], None),
    "acq_backfill": (["acquisition_backfill.py"], None),
    "taleb_tail": (["tail_index.py"], None),
    "taleb_gap": (["gap_risk.py"], None),
    "taleb_iv_skew": (["iv_skew.py", "--max-tickers", "100", "--skip-existing"], 600),
    "taleb_ergodic": (["ergodicity_ruin.py"], None),
    "taleb_fragility": (["fragility_screen.py"], None),
    "taleb_minsky": (["macro_fragility.py", "--save"], None),
    "taleb_shock": (["macro_shock.py", "--save"], None),
    "taleb_sector_shock": (["macro_sector_shock.py", "--save"], None),
    "taleb_shock_ride": (["shock_ride.py", "--save"], None),
    "taleb_arista": (["arista.py", "--save"], None),
    "taleb_ride_now": (["ride_now.py", "--save"], None),
    "taleb_subindustry_regime": (["subindustry_regime.py", "--save"], None),
    "taleb_barbell": (["barbell_check.py"], None),
    "taleb_optionality": (["hidden_optionality_audit.py"], None),
    "polygon_prices": (["update_polygon.py", "--days", "5", "--save"], 300),
    "polygon_flatfiles": (["update_polygon_flatfiles.py", "--days", "5", "--save"], 300),
    "export": (["export_dashboard_data.py"], None),
}

# dependencies: job -> set of jobs that must finish first
DEPS = {
    "rebalance": {"hmm"},
    "market_cap": set(),  # runs right after prices fetch; no deps
    "inclusion": {"preferred"},
    "implied_r": {"preferred"},
    "momentum": {"preferred"},
    "stress": {"preferred", "inclusion"},
    "risk_enrich": {"preferred"},
    "rolling": {"risk_enrich"},
    "rolling_corr": {"preferred", "risk_enrich"},
    "tail_hedge": {"rolling", "hmm"},
    "allpairs": {"preferred"},
    "screen_bt": {"preferred", "inclusion"},
    "edgar_backfill": set(),
    "backfill_new_tickers": {"edgar_backfill", "acq_backfill"},
    "fund_snap": {"edgar_backfill", "backfill_new_tickers"},
    "dupont": {"preferred"},
    "growth": {"preferred", "dupont"},
    "peer": {"preferred"},
    "earnings": {"growth", "peer"},
    "pairs": {"peer", "earnings"},
    "cross": {"peer", "earnings", "pairs"},
    "aggregate": {"cross", "earnings", "pairs", "peer", "preferred"},
    "technical": {"aggregate"},
    "econ_cal": set(),
    "est_rev": set(),
    "shadow": {"preferred", "aggregate"},
    "damodaran": {"preferred"},
    "lookthrough": {"edgar_backfill", "backfill_new_tickers"},
    "taleb_tail": {"preferred"},
    "taleb_gap": {"preferred"},
    "taleb_iv_skew": {"preferred"},
    "taleb_ergodic": {"taleb_tail"},
    "taleb_fragility": {"taleb_tail", "taleb_gap", "taleb_iv_skew"},
    "taleb_minsky": {"hmm", "taleb_fragility"},
    "taleb_shock": {"hmm"},
    "taleb_sector_shock": {"hmm"},
    "taleb_shock_ride": {"taleb_sector_shock"},
    "taleb_arista": set(),
    "taleb_subindustry_regime": {"taleb_sector_shock"},
    "taleb_barbell": {"taleb_fragility", "taleb_ergodic"},
    "taleb_optionality": {"aggregate", "preferred"},
    "hmm": set(),
    "preferred": set(),
    "crisis": set(),
    "factor_rot": set(),
    "taleb_ride_now": set(),
    "polygon_prices": set(),
    "polygon_flatfiles": set(),
    "export": {"aggregate", "technical", "econ_cal", "est_rev", "shadow",
               "taleb_tail", "taleb_gap", "taleb_iv_skew", "taleb_ergodic", "taleb_fragility", "taleb_minsky", "taleb_shock", "taleb_sector_shock", "taleb_shock_ride", "taleb_arista", "taleb_subindustry_regime", "taleb_barbell",
               "taleb_optionality"},
}

# jobs with no deps start at wave 0
def _wave(name: str, cache: dict[str, int] | None = None) -> int:
    cache = cache or {}
    if name in cache:
        return cache[name]
    d = DEPS.get(name, set())
    w = 0 if not d else 1 + max(_wave(x, cache) for x in d)
    cache[name] = w
    return w


def run_job(name: str) -> bool:
    cmd, timeout = JOBS[name]
    print(f"\n══ {name} ══")
    t0 = time.time()
    try:
        res = subprocess.run(
            [sys.executable] + cmd,
            cwd=DATA_DIR,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if res.stdout:
            for line in res.stdout.strip().splitlines()[-50:]:
                print(line)
        if res.returncode != 0:
            print(f"FAIL: {name} (exit {res.returncode})")
            if res.stderr:
                for line in res.stderr.strip().splitlines()[-20:]:
                    print(f"  stderr: {line}")
            return False
        print(f"OK {name} ({time.time() - t0:.1f}s)")
        return True
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {name} > {timeout}s")
        return False
    except Exception as e:
        print(f"ERROR: {name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Run daily analytics automation DAG")
    ap.add_argument("--only", help="comma-separated job names to run (plus deps)")
    ap.add_argument("--skip", help="comma-separated job names to skip")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="list valid jobs and exit")
    args = ap.parse_args()

    if args.list:
        for name in sorted(JOBS.keys()):
            deps = ", ".join(sorted(DEPS.get(name, set()))) or "(none)"
            print(f"  {name:<30} deps: {deps}")
        return 0

    all_names = set(JOBS.keys())
    if args.only:
        requested = {n.strip() for n in args.only.split(",") if n.strip()}
        # include deps of requested
        full = set()
        for n in requested:
            full.add(n)
            full.update(DEPS.get(n, set()))
        run_names = sorted(full, key=lambda n: _wave(n))
    else:
        run_names = sorted(all_names, key=lambda n: _wave(n))

    if args.skip:
        skip = {n.strip() for n in args.skip.split(",") if n.strip()}
        run_names = [n for n in run_names if n not in skip]

    print(f"Running {len(run_names)} jobs in wave order: {run_names}")

    # Execute in wave order; within a wave, parallel
    waves = {}
    for n in run_names:
        w = _wave(n)
        waves.setdefault(w, []).append(n)

    ok_all = True
    for w in sorted(waves.keys()):
        names = waves[w]
        if len(names) == 1:
            ok = run_job(names[0])
            ok_all = ok_all and ok
        else:
            # parallel within wave
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.max_workers, len(names))) as ex:
                fut = {ex.submit(run_job, n): n for n in names}
                for fu in concurrent.futures.as_completed(fut):
                    ok_all = ok_all and fu.result()

    if not ok_all:
        print("\n⚠ Some jobs failed (see above)")
        return 1
    print("\n✓ All jobs completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())