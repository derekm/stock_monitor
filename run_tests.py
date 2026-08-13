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


def _accel_series(n=400) -> pd.Series:
    """Price path that accelerates: log-price increases with time^1.5 (concave up),
    so 6m momentum keeps rising -> fresh acceleration at the end."""
    t = np.arange(n)
    logp = 0.5 + 1.5e-3 * t ** 1.5  # convex in time -> rising momentum
    return pd.Series(np.exp(logp), index=pd.date_range("2021-01-01", periods=n, freq="D"))


def test_breakout_detector():
    """Breakout detector: fresh breakout near high/accelerating vs off-high."""
    from breakout_detector import fresh_breakout_score
    close = _accel_series(400)
    vol = pd.Series(1e6 * (1 + np.random.default_rng(1).random(400) * 0.5), index=close.index)
    fresh = fresh_breakout_score(close, vol)
    last = fresh.iloc[-1]
    # near-high + accelerating on an accelerating series
    assert last["pth"] >= 0.90, f"accelerating series should be near its high (pth={last['pth']:.2f})"
    assert last["acceleration_ok"] == 1.0, f"rising momentum should flag acceleration ({last['acceleration_ok']})"


def test_breakout_verdict_distinguishes():
    """FRESH_BREAKOUT vs NO_SIGNAL: accelerating-vs-flat series differ."""
    from breakout_detector import fresh_breakout_score
    idx = pd.date_range("2021-01-01", periods=300, freq="D")
    up = _accel_series(300)  # accelerating up
    # deterministic range-bound: sine wave, ends mid-range (not near a fresh high)
    t = np.arange(300)
    flat = pd.Series(100 + 10 * np.sin(t / 20.0), index=idx)
    vu = fresh_breakout_score(up, None)
    vf = fresh_breakout_score(flat, None)
    assert vu.iloc[-1]["verdict"] in ("FRESH_BREAKOUT", "BUILDING"), f"up-series should be fresh/building, got {vu.iloc[-1]['verdict']}"
    assert vf.iloc[-1]["verdict"] != "FRESH_BREAKOUT", "flat-series should not be a fresh breakout"


def test_fractal_posture_distinguishes():
    """fractal_posture: BROAD (multi-view confirmed) vs NARROW vs MIXED."""
    from fractal_windows import fractal_multi_view, fractal_posture
    # broad: strongly accelerating series -> both views' best spans confirm
    broad = _accel_series(400)
    mv = fractal_multi_view(broad, configs=[(30, 3), (10, 3)])
    p = fractal_posture(mv)
    assert p["posture"] in ("BROAD", "MIXED"), f"accelerating series should be broad/mixed, got {p['posture']}"
    assert p["n_confirmed"] >= 1, f"should confirm at least one view ({p['n_confirmed']})"
    # weak: slowly-declining / flat series -> best spans should NOT confirm
    t = np.arange(400)
    flat = pd.Series(100 - 0.02 * t + 2 * np.sin(t / 40.0),
                     index=pd.date_range("2021-01-01", periods=400, freq="D"))
    mv2 = fractal_multi_view(flat, configs=[(30, 3), (10, 3)])
    p2 = fractal_posture(mv2)
    assert p2["posture"] in ("WEAK", "NARROW", "MIXED"), f"declining series should not be broad, got {p2['posture']}"


def test_ride_gate_opens_on_quality_not_history():
    """ride_gate opens on short history when quality (stack/durability) is high."""
    from ride_longevity import ride_gate
    idx = pd.date_range("2026-01-01", periods=4, freq="ME")
    m = pd.Series([0.05, 0.06, 0.07, 0.08], index=idx)  # 4mo, strong up
    g = ride_gate(m, stack_depth=4, long_ride=0.5, reliability="building")
    assert g["gate_open"], f"4mo strong quality should open gate ({g['reasons']})"
    g2 = ride_gate(m, stack_depth=0, long_ride=0.2)
    assert not g2["gate_open"], "4mo weak quality should stay closed"
    assert "quality_too_low" in g2["reasons"]


def test_ride_exit_holds_pullback_exits_breakdown():
    """ride_exit holds a dip when stack is intact, exits when stack breaks."""
    from ride_longevity import ride_exit
    idx = pd.date_range("2026-01-01", periods=5, freq="ME")
    # 3m rollover (negative recent months)
    m = pd.Series([0.06, 0.07, -0.04, -0.02, -0.01], index=idx)
    e_hold = ride_exit(m, stack_depth=4, long_ride=0.5)     # stack intact
    assert not e_hold["exit"], f"dip with strong stack should hold, got {e_hold['reasons']}"
    e_exit = ride_exit(m, stack_depth=0, long_ride=0.2)     # stack broke
    assert e_exit["exit"], f"dip with broken stack should exit ({e_exit['reasons']})"
    assert e_exit["exit_kind"] == "rollover_confirm"


def test_long_ride_score_finite_and_discriminates():
    """long_ride_score is finite (even with NaN volume) and separates up vs flat."""
    from ride_longevity import long_ride_score
    idx = pd.date_range("2021-01-01", periods=400, freq="D")
    up = _accel_series(400)
    vol = pd.Series(1e6 + np.arange(400) * 1e4, index=idx)
    d = long_ride_score(up, vol)
    assert pd.notna(d["long_ride_score"].iloc[-1]), "score must be finite"
    # declining series should score lower on pullback resilience
    t = np.arange(400)
    flat = pd.Series(100 - 0.03 * t, index=idx)
    df_ = long_ride_score(flat, None)
    assert df_["long_ride_score"].iloc[-1] < d["long_ride_score"].iloc[-1], \
        "durable up-trend should outscore a declining series"


def test_momentum_stack_series_orders_short_to_long():
    """momentum_stack_series gives per-date depth, ordered short->long."""
    from fractal_windows import fractal_multi_view, momentum_stack_series
    up = _accel_series(400)
    mv = fractal_multi_view(up, configs=[(5, 3), (10, 3), (15, 3), (30, 3)])
    s = momentum_stack_series(mv)
    assert "stack_depth" in s.columns and "full_stack" in s.columns
    assert s["stack_depth"].max() >= 1, "accelerating series should build a stack"
    assert (s["stack_depth"] <= 4).all(), "depth bounded by n_views"
    assert (s["stack_depth"].dtype == int or s["stack_depth"].dtype == np.int64)


def test_structural_gate_modes_run_and_discriminate():
    """structural_gate modes produce valid positions on an uptrend vs a flat series."""
    from ride_longevity import structural_positions, structural_gate, STRUCTURAL_MODES
    idx = pd.date_range("2021-01-01", periods=400, freq="D")
    up = _accel_series(400)
    up.index = idx
    t = np.arange(400)
    flat = pd.Series(100 - 0.02 * t, index=idx)
    for mode in STRUCTURAL_MODES:
        p_up = structural_positions(up, mode=mode)
        assert len(p_up) == 400, f"{mode}: position length"
        assert (p_up >= 0).all(), f"{mode}: positions non-negative"
        g = structural_gate(up, mode=mode)
        assert "gate_open" in g and "signal" in g, f"{mode}: gate dict"
    # an accelerating series should produce higher mean position than a declining one
    # for the trend-following modes (regime / recouple / momentum)
    for mode in ["regime", "recouple", "momentum", "volscale"]:
        p_up = structural_positions(up, mode=mode)
        p_flat = structural_positions(flat, mode=mode)
        assert p_up.mean() >= p_flat.mean(), f"{mode}: uptrend should hold more than declining"


def test_statistical_profiler_wide_and_finite():
    """statistical_profiler emits the full wide stat set, finite, and sane."""
    from statistical_profiler import window_profile_stats, STAT_COLS, profile_ticker
    idx = pd.date_range("2021-01-01", periods=300, freq="D")
    c = _accel_series(300)
    c.index = idx
    vol = pd.Series(1e6 + np.arange(300) * 1e4, index=idx)
    df = window_profile_stats(c, vol, 30)
    assert list(df.columns) == ["close"] + STAT_COLS, "profiler must emit close + documented stat columns"
    last = df.iloc[-1]
    # an accelerating series ends at/near its window high -> runup high, pctile high
    assert pd.notna(last["price_mean"]) and last["price_mean"] > 0
    assert last["runup"] > 0.5, f"accelerating series should end high in range (runup {last['runup']:.2f})"
    assert last["close_pctile"] > 0.5, f"accelerating series should be at high percentile (pctile {last['close_pctile']:.2f})"
    assert pd.notna(last["vwap"]) and last["vwap"] > 0
    assert pd.notna(last["price_skew"]) and pd.notna(last["price_kurtosis"])
    assert pd.notna(last["momentum"]) and pd.notna(last["log_ret"])
    # median/mean/mode positive
    assert last["price_median"] > 0 and last["price_mode"] > 0
    # window drawdown <= 0 always
    assert (df["window_drawdown"].dropna() <= 1e-9).all(), "drawdown must be non-positive"
    # profile_ticker long format
    pf = profile_ticker(c, vol, [(0, 30, 30), (0, 60, 60)])
    assert "ticker" not in pf.columns  # no ticker yet (added by caller)
    assert set(["date", "span_from", "span_to", "span_len", "close"]).issubset(pf.columns)
    assert pf["span_len"].nunique() == 2

    # true-OHLCV stats: populated when OHLC given, NaN when not
    o = c * 0.99; h = c * 1.01; lo = c * 0.98
    df_ohlc = window_profile_stats(c, vol, 30, open_=o, high=h, low=lo)
    for col in ["vwap_true", "atr", "atr_pct", "gap_mean", "gap_std",
                "range_hl", "body_mean", "body_std", "upper_wick", "lower_wick"]:
        assert pd.notna(df_ohlc[col].iloc[-1]), f"{col} should be finite with OHLC"
        assert df[col].isna().all(), f"{col} should be all-NaN without OHLC"
    # true VWAP = window Σ(typical·vol)/Σvol, so it lies within the window's
    # high/low range (typical price (H+L+C)/3 is bounded by H and L)
    last = df_ohlc.iloc[-1]
    assert df_ohlc["vwap_true"].dropna().between(
        df_ohlc["price_min"].dropna(), df_ohlc["price_max"].dropna()).all(), \
        "true VWAP within window price range"
    assert last["atr"] >= 0, "ATR non-negative"


# ── registry ────────────────────────────────────────────────────────────
TESTS = {
    "spans": test_spans_generator,
    "tsmom": test_tsmom_signal,
    "young_gate": test_young_gate_reliability,
    "fractal_vec": test_fractal_signal_vectorized_identity,
    "cpu_gpu": test_fractal_cpu_gpu_concur,
    "momentum_schema": test_momentum_parquet_schema,
    "research_report": test_research_report_keys,
    "breakout_detector": test_breakout_detector,
    "breakout_verdict": test_breakout_verdict_distinguishes,
    "fractal_posture": test_fractal_posture_distinguishes,
    "ride_gate": test_ride_gate_opens_on_quality_not_history,
    "ride_exit": test_ride_exit_holds_pullback_exits_breakdown,
    "long_ride": test_long_ride_score_finite_and_discriminates,
    "stack_series": test_momentum_stack_series_orders_short_to_long,
    "structural_gate": test_structural_gate_modes_run_and_discriminate,
    "statistical_profiler": test_statistical_profiler_wide_and_finite,
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
