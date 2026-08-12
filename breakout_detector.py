#!/usr/bin/env python3
"""breakout_detector.py — FRESH breakout detection, layered on the fractal consensus.

Implements the composite fresh-breakout detector recommended by the research
audit (George-Hwang 2004, Gettleman-Marks acceleration, Donchian, volume/
exhaustion). Distinguishes a NEW breakout from a maturing/exhausted one and
avoids buying the top. Runs on top of the fractal-of-sliding-windows momentum
(fractal_windows.py) so it also has the multi-granularity agreement view.

Fresh-breakout composite (higher = fresher):
  1. PTH — price-to-52-week-high ratio (George-Hwang 2004): nearness to the
     prior high predicts returns that DON'T reverse. Fresh = price pushing
     through / near the long-term high.
  2. Donchian close-break — close beyond the trailing N-day high on the longest
     window (a true close-through, not a first-tick wick), confirming the break.
  3. ACCELERATION — 2nd derivative: 6-month momentum positive AND month-over-
     month momentum rising (Gettleman-Marks 2006): high + rising momentum beats
     high momentum alone. The key 'freshness' discriminator.
  4. VOLUME — multi-day volume expansion with rising OBV (institutional
     participation).

Rejection / maturity flags (score as 'maturing', not fresh):
  - decelerating momentum (2nd derivative negative)
  - volume divergence (price up, volume down / OBV flat-declining)
  - exhaustion gap / blow-off (extreme single-bar volume + wide range)
  - parabolic volatility expansion

All functions pure (operate on close+volume Series). No data I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fractal_windows import fractal_signal_vec, fractal_consensus


# ── 1. Price-to-52-week-high (George-Hwang 2004) ────────────────────────
def pth(close: pd.Series, window: int = 252) -> pd.Series:
    """Price-to-prior-high ratio = close[t] / max(close[t-window..t])."""
    hi = close.rolling(window, min_periods=1).max()
    return close / hi


# ── 2. Donchian close-break (longest window) ────────────────────────────
def donchian_break(close: pd.Series, window: int = 60) -> pd.Series:
    """True close beyond the trailing N-day high (close-based Donchian upper).
    1.0 when close[t] == max(close[t-window..t]) AND it's a new max (not the
    first day of the window)."""
    hi = close.rolling(window, min_periods=1).max()
    prior_hi = close.rolling(window, min_periods=1).max().shift(1)
    return ((close >= hi) & (close > prior_hi)).astype(float)


# ── 3. Acceleration / momentum-of-momentum (Gettleman-Marks) ────────────
def momentum(close: pd.Series, months: int = 6) -> pd.Series:
    """Trailing months-month return (log), resampled to business-month ends."""
    logc = np.log(close)
    return logc - logc.shift(months * 21)


def acceleration(close: pd.Series, months: int = 6) -> pd.Series:
    """2nd derivative: change in momentum. Positive = momentum rising = fresh.
    Returns a Series of the same index = momentum[t] - momentum[t-m]."""
    m = momentum(close, months)
    return m - m.shift(months * 21)


# ── 4. Volume confirmation ──────────────────────────────────────────────
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative volume signed by price direction."""
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()


def volume_expansion(volume: pd.Series, window: int = 10) -> pd.Series:
    """Multi-day volume above its trailing average (sustained expansion)."""
    avg = volume.rolling(window, min_periods=1).mean()
    return volume / avg


# ── composite fresh-breakout score ──────────────────────────────────────
def fresh_breakout_score(close: pd.Series, volume: pd.Series | None = None,
                         *, pth_window: int = 252, donch_window: int = 60,
                         mom_months: int = 6) -> pd.DataFrame:
    """Composite fresh-breakout score per date.

    close: DatetimeIndexed price. volume: optional DatetimeIndexed volume.
    Returns a DataFrame indexed by date with components + composite + verdict.
    """
    p = pth(close, pth_window)
    db = donchian_break(close, donch_window)
    m6 = momentum(close, mom_months)
    acc = acceleration(close, mom_months)
    mom_ok = m6 > 0
    acc_ok = acc > 0  # momentum rising (2nd derivative positive)
    fresh_acc = (mom_ok & acc_ok).astype(float)  # positive AND rising

    out = pd.DataFrame({
        "pth": p,
        "donchian_break": db,
        "mom_6m": m6,
        "acceleration": acc,
        "acceleration_ok": acc_ok.astype(float),
        "mom_ok": mom_ok.astype(float),
        "fresh_acceleration": fresh_acc,
    })

    if volume is not None:
        o = obv(close, volume)
        o_rising = (o.diff(20) > 0).astype(float)
        vexp = (volume_expansion(volume, 10) > 1.0).astype(float)  # above 10d avg
        vol_ok = (o_rising * vexp).astype(float)
        out["obv_rising"] = o_rising
        out["volume_expanding"] = vexp
        out["volume_confirmed"] = vol_ok
        # exhaustion: price high but OBV falling = volume divergence
        out["volume_divergence"] = (fresh_acc.astype(float) * (1 - o_rising)).astype(float)
    else:
        out["obv_rising"] = np.nan
        out["volume_expanding"] = np.nan
        out["volume_confirmed"] = np.nan
        out["volume_divergence"] = np.nan

    # composite: weight PTH + Donchian break + fresh acceleration + volume
    comp = (0.25 * out["pth"].fillna(0) +
            0.20 * out["donchian_break"] +
            0.35 * out["fresh_acceleration"] +
            0.20 * out["volume_confirmed"].fillna(0.5))
    out["fresh_score"] = comp

    # verdict
    def verdict(r):
        if r["fresh_acceleration"] and r["pth"] >= 0.90:
            return "FRESH_BREAKOUT"
        if r["acceleration_ok"] and r["pth"] < 0.90:
            return "BUILDING"
        if not r["acceleration_ok"] and r["pth"] >= 0.90:
            return "MATURING"
        if r["volume_divergence"] and r["pth"] >= 0.90:
            return "EXHAUSTED"
        return "NO_SIGNAL"
    out["verdict"] = out.apply(verdict, axis=1)
    return out


def fractal_fresh(close: pd.Series, volume: pd.Series | None = None,
                  *, a: int = 30, b: int = 3) -> pd.DataFrame:
    """Combine the fractal consensus (multi-granularity uptrend agreement) with
    the fresh-breakout components. Returns a per-date DataFrame joining both."""
    fdf = fractal_signal_vec(close, a, b)
    cons = fractal_consensus(fdf)
    fresh = fresh_breakout_score(close, volume)
    joined = cons.join(fresh, how="inner")
    # a 'fresh + confirmed' breakout = fractal agreement AND fresh acceleration
    joined["fresh_confirmed"] = (
        (joined["frac_uptrend"] >= 0.6) & (joined["fresh_acceleration"] == 1.0)
    ).astype(int)
    return joined
