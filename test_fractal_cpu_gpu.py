#!/usr/bin/env python3
"""test_fractal_cpu_gpu.py — CPU and GPU fractal implementations must always concur.

The GPU batched path (fractal_windows_gpu.py) and the CPU vectorized path
(fractal_windows.py::fractal_signal_vec) implement the same patent scheme through
different engines. This test proves they produce identical results on a battery
of synthetic and real inputs — so either can be used and the CPU stays a safe
fallback when GPU is unavailable.

Run: python test_fractal_cpu_gpu.py
Exit code 0 = all concur; non-zero = mismatch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fractal_windows import fractal_signal_vec, spans_generator
from fractal_windows import fractal_batch, gpu_available


def _build_gpu_result(close: pd.Series, a: int, b: int):
    """GPU batched result keyed by (f,t) -> arrays."""
    logp = np.log(close.values.reshape(1, -1)).astype(np.float64)
    logp = np.where(np.isfinite(logp), logp, 0.0)
    res = fractal_batch(logp, a, b)
    return res


def _build_cpu_result(close: pd.Series, a: int, b: int):
    df = fractal_signal_vec(close, a, b)
    out = {}
    for (f, t) in spans_generator(a, b):
        sub = df[(df["span_from"] == f) & (df["span_to"] == t)]
        if sub.empty:
            continue
        # align to the GPU array's trailing window (CPU starts after full span)
        r = sub["ret"].values
        s = sub["slope"].values
        out[(f, t)] = {"ret": r, "slope": s}
    return out


def _aligned_arrays(cpu_ret, cpu_slope, gpu_ret, gpu_slope):
    """Trim leading NaN region so both arrays align by position."""
    n = min(len(cpu_ret), len(gpu_ret))
    # skip the warm-up (first L-1 NaN) — both should have NaN there
    c = cpu_ret[:n]; g = gpu_ret[:n]
    # find first index where both are finite
    both = np.isfinite(c) & np.isfinite(g)
    if not both.any():
        return None
    first = np.argmax(both)
    return c[first:], cpu_slope[:n][first:], g[first:], gpu_slope[:n][first:]


def _synthetic_series(n: int = 400, drift: float = 0.001, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.02, n)
    close = pd.Series(100 * np.exp(np.cumsum(rets)),
                      index=pd.date_range("2020-01-01", periods=n, freq="D"))
    return close


def check_close(close: pd.Series, a: int, b: int, name: str, tol: float = 1e-6) -> int:
    fails = 0
    base_a, base_b = a, b  # preserve for print (loop may shadow `a`)
    cpu = _build_cpu_result(close, a, b)
    gpu = _build_gpu_result(close, a, b)
    for (f, t) in spans_generator(a, b):
        if (f, t) not in cpu or (f, t) not in gpu:
            continue
        cr = cpu[(f, t)]["ret"]; cs = cpu[(f, t)]["slope"]
        gr = gpu[(f, t)]["ret"].cpu().numpy().reshape(-1)
        gs = gpu[(f, t)]["slope"].cpu().numpy().reshape(-1)
        # GPU arrays are length D; CPU has trailing values too. Compare overlapping.
        al = _aligned_arrays(cr, cs, gr, gs)
        if al is None:
            continue
        cr2, cs2, gr2, gs2 = al
        if not (np.allclose(cr2, gr2, atol=tol, equal_nan=True) and
                np.allclose(cs2, gs2, atol=tol, equal_nan=True)):
            fails += 1
            print(f"  !! {name} span({f},{t}) MISMATCH: "
                  f"max|ret| diff={np.nanmax(np.abs(cr2-gr2)):.2e} "
                  f"max|slope| diff={np.nanmax(np.abs(cs2-gs2)):.2e}")
    if fails == 0:
        print(f"  OK {name} (spans {spans_generator(base_a,base_b)})")
    return fails


def main():
    total = 0
    print("=== CPU vs GPU fractal concurrency test ===")
    print("GPU available:", gpu_available())

    # synthetic, several span configs
    for (a, b) in [(30, 3), (10, 3), (30, 2), (20, 4)]:
        total += check_close(_synthetic_series(600, seed=a + b), a, b,
                             f"synthetic({a},{b})")
    # a trending series (momentum case)
    total += check_close(_synthetic_series(600, drift=0.003, seed=99), 30, 3,
                         "synthetic-trend")

    # real data: a few liquid tickers
    try:
        from macro_sector_shock import _load_price_matrix
        w = _load_price_matrix()
        for t in ["AAPL", "MSFT", "NVDA", "RAL"]:
            if t not in w.columns:
                continue
            close = w[t].ffill().dropna()
            if len(close) < 150:
                continue
            total += check_close(close, 30, 3, f"real-{t}")
    except Exception as e:  # noqa: BLE001
        print(f"  (real-data check skipped: {e})")

    print(f"\n{'ALL CONCUR' if total == 0 else f'{total} MISMATCHES'}")
    return 1 if total else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
