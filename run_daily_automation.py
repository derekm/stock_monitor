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
import shutil
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

# Jobs run WITHOUT timeouts (None = wait forever). These are multi-hour
# pipelines (update_prices yfinance fallback, export, the Vulkan LLM stages);
# a timeout kills them mid-write and leaves a partial wave that retries the
# same slow job next run. Failfast is not worth torn outputs — let jobs
# finish. Keep DEFAULT_JOB_TIMEOUT = None.
DEFAULT_JOB_TIMEOUT = None


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
        # Per-job interpreter override. Everything downstream of Docling
        # (mentions, press, notes, llm_forecast) is pinned to the Vulkan
        # .venv-xpu; ingest/docling/indicators stay on the runner default.
        py = spec.get("python") or sys.executable
        timeout = spec.get("timeout")
        if timeout is not None:
            timeout = int(timeout)
        else:
            # 43 of 55 jobs shipped with `timeout: null`, i.e. wait forever. One
            # wedged job then blocks its whole wave and everything downstream
            # never runs — which is how rolling_corr / cross_asset_stability
            # went 15 days stale with no failure surfaced. An unset timeout now
            # gets a ceiling; explicit per-job values still win.
            timeout = DEFAULT_JOB_TIMEOUT
        jobs[name] = (py, cmd, timeout)
    
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
    py, cmd, timeout = JOBS[name]
    print(f"\n══ {name} ══", flush=True)
    t0 = time.time()
    try:
        # Disk preflight: several jobs (crisis_correlation, export) write
        # 0.5-2 GB intermediates and OSError 112 mid-job if the drive fills.
        # Failing loudly here beats failing at step 2000 of a 20-minute job.
        free_gb = shutil.disk_usage(DATA_DIR).free / 1e9
        if free_gb < 12:
            print(f"WARNING: only {free_gb:.1f} GB free on {DATA_DIR} — "
                  f"heavy jobs (crisis, export, bogle) may OSError 112", flush=True)
        # -u on the child and flush=True on every parent print: when this runner
        # is redirected to a file (nohup / start /B / cron), block buffering made
        # the log look empty for the whole run — dag_remainder.log sat at 42 bytes
        # while jobs were executing, so a stalled run was indistinguishable from
        # a dead one.
        res = subprocess.run(
            [py, "-u"] + cmd,
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
            print("", flush=True)
            return False
        print(f"OK {name} ({time.time() - t0:.1f}s)", flush=True)
        return True
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {name} > {timeout}s", flush=True)
        return False
    except Exception as e:
        print(f"ERROR: {name}: {e}", flush=True)
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
        unknown = requested - all_names
        if unknown:
            print(f"ERROR: unknown job(s): {sorted(unknown)}")
            print("Run --list to see valid job names.")
            return 2
        # Include the TRANSITIVE closure of deps, not just direct ones.
        # `--only cross` used to pull {earnings, pairs, peer} and silently omit
        # {dupont, growth, preferred}, so cross ran at wave 5 against upstream
        # outputs that were never built in this invocation.
        full = set()
        stack = list(requested)
        while stack:
            n = stack.pop()
            if n in full:
                continue
            full.add(n)
            stack.extend(DEPS.get(n, set()) - full)
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
    failed: set[str] = set()
    skipped: set[str] = set()
    results: dict[str, str] = {}
    scheduled = set(run_names)

    for w in sorted(waves.keys()):
        names = waves[w]
        # Do not run a job whose upstream failed or was skipped: it would read a
        # stale/missing parquet and "succeed", and `export` (27 deps) would then
        # publish a dashboard built on last week's data with a green summary.
        blocked = {}
        runnable = []
        for n in names:
            bad = (DEPS.get(n, set()) & (failed | skipped)) & scheduled
            if bad:
                blocked[n] = sorted(bad)
            else:
                runnable.append(n)
        for n, bad in blocked.items():
            print(f"\n══ {n} ══\nSKIP: upstream not OK: {bad}")
            skipped.add(n)
            results[n] = f"SKIPPED (upstream {','.join(bad)})"
            ok_all = False

        if not runnable:
            continue
        if len(runnable) == 1:
            ok = run_job(runnable[0])
            results[runnable[0]] = "OK" if ok else "FAILED"
            if not ok:
                failed.add(runnable[0])
            ok_all = ok_all and ok
        else:
            # parallel within wave
            with ThreadPoolExecutor(max_workers=min(args.max_workers, len(runnable))) as ex:
                fut = {ex.submit(run_job, n): n for n in runnable}
                for fu in as_completed(fut):
                    n = fut[fu]
                    ok = fu.result()
                    results[n] = "OK" if ok else "FAILED"
                    if not ok:
                        failed.add(n)
                    ok_all = ok_all and ok

    # Explicit run summary: a 55-job run scrolls its own failures off screen,
    # and the old ending printed only a one-line "Some jobs failed (see above)".
    print(f"\n{'='*60}\nRun summary: {len(results)} jobs")
    for n in sorted(results, key=lambda x: _wave(x)):
        if results[n] != "OK":
            print(f"  {results[n]:<34} {n}")
    n_ok = sum(1 for v in results.values() if v == "OK")
    print(f"  OK: {n_ok}  FAILED: {len(failed)}  SKIPPED: {len(skipped)}")

    if not ok_all:
        if failed:
            print(f"\n⚠ FAILED: {sorted(failed)}")
        if skipped:
            print(f"⚠ SKIPPED (upstream): {sorted(skipped)}")
        return 1
    print("\n✓ All jobs completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())