#!/usr/bin/env python3
"""
run_daily_automation.py — Master daily job for stock_monitor analytics stack.

Runs 40+ jobs as a dependency DAG with multiprocessing (independent jobs run
in parallel; only real dependencies serialize). Falls back to sequential
execution if multiprocessing is unavailable. Universe is daily_prices.

DAG configuration is loaded from daily_automation_dag.yaml.

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
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

DATA_DIR = Path(__file__).parent
DAG_FILE = DATA_DIR / "daily_automation_dag.yaml"


def load_dag() -> tuple[dict, dict]:
    """Load JOBS and DEPS from YAML file (single source of truth)."""
    if not HAS_YAML:
        raise RuntimeError("PyYAML not installed. Install with: pip install pyyaml")
    
    if not DAG_FILE.exists():
        raise FileNotFoundError(f"DAG file not found: {DAG_FILE}")
    
    with open(DAG_FILE) as f:
        dag = yaml.safe_load(f)
    
    jobs = {}
    for name, spec in dag.get("jobs", {}).items():
        cmd = spec.get("cmd", [])
        timeout = spec.get("timeout")
        if timeout is not None:
            timeout = int(timeout)
        jobs[name] = (cmd, timeout)
    
    deps = {}
    for name, dep_list in dag.get("dependencies", {}).items():
        deps[name] = set(dep_list)
    
    # Ensure all jobs have dep entries
    for name in jobs:
        if name not in deps:
            deps[name] = set()
    
    print(f"Loaded DAG from {DAG_FILE}: {len(jobs)} jobs")
    return jobs, deps


# Load DAG
JOBS, DEPS = load_dag()

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
            with ThreadPoolExecutor(max_workers=min(args.max_workers, len(names))) as ex:
                fut = {ex.submit(run_job, n): n for n in names}
                for fu in as_completed(fut):
                    ok_all = ok_all and fu.result()

    if not ok_all:
        print("\n⚠ Some jobs failed (see above)")
        return 1
    print("\n✓ All jobs completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())