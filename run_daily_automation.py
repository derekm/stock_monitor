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
    "hmm": (["hmm_regime_detection.py", "--n-states", "3", "--save"], 300),
    "rebalance": (["rebalance_calendar.py", "--months", "18", "--save"], 300),
    "preferred": (["preferred_metrics.py", "--save"], 300),
    "inclusion": (["inclusion_criteria.py", "--explore-defensive", "--save"], 300),
    "stress": (["stress_dual_pass.py", "--save"], 300),
    "crisis": (["crisis_correlation.py", "--save"], 600),
    "factor_rot": (["factor_rotation_defense.py", "--save"], 300),
    "risk_enrich": (["risk_enrich.py"], 300),
    "rolling": (["rolling_window_analysis.py", "--universe", "portfolio", "--save"], 300),
    "rolling_corr": (["rolling_correlation_windows.py", "--save"], 300),
    "tail_hedge": (["tail_risk_hedging.py", "--save"], 300),
    "allpairs": (["allpairs_correlations.py", "--window", "63", "--step", "21", "--max-assets", "50"], 600),
    "fund_snap": (["fundamentals_history.py", "snapshot"], 300),
    "screen_bt": (["fundamentals_history.py", "backtest-screens"], 300),
    "dupont": (["dupont_analysis.py", "--save"], 300),
    "growth": (["growth_tech_analytics.py"], 600),
    "peer": (["peer_analytics.py", "--save"], 900),
    "earnings": (["earnings_catalyst.py", "--save"], 300),
    "pairs": (["pair_engine.py", "--save"], 900),
    "cross": (["cross_section.py", "--save"], 600),
    "aggregate": (["signal_aggregator.py", "--save"], 300),
    "technical": (["technical_signals.py", "--save"], 600),
    "econ_cal": (["economic_calendar.py", "--save"], 120),
    "est_rev": (["estimate_revisions.py", "--save"], 600),
    "shadow": (["shadow_book.py", "--save"], 300),
    "taleb_tail": (["tail_index.py"], 600),
    "taleb_gap": (["gap_risk.py"], 600),
    "taleb_ergodic": (["ergodicity_ruin.py"], 900),
    "taleb_fragility": (["fragility_screen.py"], 600),
    "taleb_minsky": (["macro_fragility.py", "--save"], 300),
    "taleb_shock": (["macro_shock.py", "--save"], 300),
    "taleb_barbell": (["barbell_check.py"], 600),
    "taleb_optionality": (["hidden_optionality_audit.py"], 600),
    "export": (["export_dashboard_data.py"], 600),
}

# dependencies: job -> set of jobs that must finish first
DEPS = {
    "rebalance": {"hmm"},
    "inclusion": {"preferred"},
    "stress": {"preferred", "inclusion"},
    "risk_enrich": {"preferred"},
    "rolling": {"risk_enrich"},
    "rolling_corr": {"preferred", "risk_enrich"},
    "tail_hedge": {"rolling", "hmm"},
    "allpairs": {"preferred"},
    "screen_bt": {"preferred", "inclusion"},
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
    "taleb_tail": {"preferred"},
    "taleb_gap": {"preferred"},
    "taleb_ergodic": {"taleb_tail"},
    "taleb_fragility": {"taleb_tail", "taleb_gap"},
    "taleb_minsky": {"hmm", "taleb_fragility"},
    "taleb_shock": {"hmm"},
    "taleb_barbell": {"taleb_fragility", "taleb_ergodic"},
    "taleb_optionality": {"aggregate", "preferred"},
    "export": {"aggregate", "technical", "econ_cal", "est_rev", "shadow",
               "taleb_tail", "taleb_gap", "taleb_ergodic", "taleb_fragility", "taleb_minsky", "taleb_shock", "taleb_barbell",
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
        r = subprocess.run(
            [sys.executable, *cmd],
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
    except Exception as e:  # noqa: BLE001
        print(f"ERROR {name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of job names")
    ap.add_argument("--sequential", action="store_true", help="force sequential (debug)")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else set(JOBS)
    only = {j for j in only if j in JOBS}
    if not only:
        raise SystemExit("no valid jobs selected")

    waves: dict[int, list[str]] = {}
    for j in only:
        w = _wave(j)
        # include dependency chain when not --only
        if args.only:
            for dep in DEPS.get(j, set()):
                if dep in only:
                    continue
        waves.setdefault(w, []).append(j)
    order = sorted(waves)

    results: dict[str, bool] = {}
    t_all = time.time()
    if args.sequential or len(order) == 1:
        for w in order:
            for j in waves[w]:
                results[j] = run_job(j)
    else:
        import concurrent.futures
        for w in order:
            batch = waves[w]
            if len(batch) == 1:
                results[batch[0]] = run_job(batch[0])
                continue
            print(f"\n── wave {w}: parallel {batch} ──")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as ex:
                futs = {ex.submit(run_job, j): j for j in batch}
                for f in concurrent.futures.as_completed(futs):
                    results[futs[f]] = f.result()

    print(f"\n══ SUMMARY ({time.time() - t_all:.0f}s) ══")
    for k, v in results.items():
        print(f"  {'✓' if v else '✗'} {k}")
    failed = [k for k, v in results.items() if not v]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
