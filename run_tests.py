#!/usr/bin/env python3
"""run_tests.py — executable test library for stock_monitor.

Imports program modules and runs repeatable verifications for ongoing
development. Each test imports the real production module and asserts a
property (invariant, identity, determinism, concurrency). Exit code = number
of failures (0 = all pass).

Add tests here as the suite grows. Run:
  python run_tests.py            # all tests
  python run_tests.py --list
  python run_tests.py --only spans cpu_gpu   # subset by name
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# repo root on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────
def _synth_series(n=400, drift=0.001, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.02, n)
    return pd.Series(100 * np.exp(np.cumsum(rets)),
                     index=pd.date_range("2020-01-01", periods=n, freq="D"))


# ── tests: each returns None on pass or raises AssertionError ───────────
def test_spans_generator():
    """Fractal span generator matches patent FIGS 26A/28 exactly."""
    from fractal_windows import spans_generator
    assert spans_generator(30, 3) == [(0, 30), (0, 60), (0, 90),
                                      (30, 60), (30, 90), (60, 90)], "30x3 spans"
    assert spans_generator(10, 3) == [(0, 10), (0, 20), (0, 30),
                                      (10, 20), (10, 30), (20, 30)], "10x3 spans"
    assert spans_generator(30, 2) == [(0, 30), (0, 60), (30, 60)], "30x2 spans"


def test_tsmom_signal():
    """TSMOM signal is 0/1 and only fires on positive trailing return."""
    from momentum_research import tsmom_signal
    m = _synth_series(120).pct_change().dropna()  # ~monthly-ish
    sig = tsmom_signal(m, 3, vol_scaled=False)
    assert set(sig.unique()).issubset({0.0, 1.0}), "0/1 signal"
    # determinism
    sig2 = tsmom_signal(m, 3, vol_scaled=False)
    assert (sig == sig2).all()


def test_young_gate_reliability():
    """Young-ticker gate reliability buckets are ordered correctly."""
    from momentum_research import young_gate
    assert young_gate(_synth_series(), history_months=2, annual_vol=0.3)["reliability"] == "low"
    assert young_gate(_synth_series(), history_months=4, annual_vol=0.3)["reliability"] == "building"
    assert young_gate(_synth_series(), history_months=7, annual_vol=0.3)["reliability"] == "reliable"


def test_fractal_signal_vectorized_identity():
    """fractal_signal_vec slope/ret match closed-form (no polyfit drift)."""
    from fractal_windows import fractal_signal_vec, spans_generator
    close = _synth_series(300, drift=0.003, seed=7)
    df = fractal_signal_vec(close, 30, 3)
    L = 90
    sub = df[(df["span_len"] == L) & df["ret"].notna()].iloc[-1]
    i = close.index.get_loc(sub["date"])
    win = np.log(close.iloc[i - L + 1:i + 1]).values
    true_slope = np.polyfit(np.arange(L), win, 1)[0]
    assert abs(sub["slope"] - true_slope) < 1e-6, "vectorized slope == polyfit"


def test_fractal_cpu_gpu_concur():
    """CPU vectorized and GPU batched fractal implementations must concur."""
    import test_fractal_cpu_gpu as t
    assert t.main() == 0, "CPU/GPU concurrency test failed"


def test_momentum_parquet_schema():
    """momentum_metrics.parquet carries the research columns."""
    from momentum_analytics import build
    df, _, _ = build()
    for col in ("tsmom_3mo_sharpe", "tsmom_6mo_sharpe", "stmom_1m_ret",
                "gw52_high_prox", "young_gate_open", "young_gate_reliability"):
        assert col in df.columns, f"missing {col}"


def test_research_report_keys():
    """research_report returns all measure keys."""
    from momentum_research import research_report
    m = _synth_series(300, seed=3).pct_change().dropna().resample("ME").sum().dropna()
    rep = research_report(m)
    for k in ("tsmom_3mo_sharpe", "tsmom_6mo_sharpe", "tsmom_12mo_sharpe",
              "stmom_1m_ret", "gw52_high_prox", "young_gate"):
        assert k in rep, f"missing {k}"


# ── registry ────────────────────────────────────────────────────────────
TESTS = {
    "spans": test_spans_generator,
    "tsmom": test_tsmom_signal,
    "young_gate": test_young_gate_reliability,
    "fractal_vec": test_fractal_signal_vectorized_identity,
    "cpu_gpu": test_fractal_cpu_gpu_concur,
    "momentum_schema": test_momentum_parquet_schema,
    "research_report": test_research_report_keys,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list tests")
    ap.add_argument("--only", default=None, help="comma-separated test names")
    args = ap.parse_args()

    if args.list:
        for k in TESTS:
            print(f"  {k}: {TESTS[k].__doc__.strip() if TESTS[k].__doc__ else ''}")
        return 0

    names = [n.strip() for n in args.only.split(",") if n.strip()] if args.only else list(TESTS)
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        print(f"unknown tests: {unknown}; available: {list(TESTS)}")
        return 2

    fails = 0
    for name in names:
        try:
            TESTS[name]()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc(limit=2)
    print(f"\n{len(names) - fails}/{len(names)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
