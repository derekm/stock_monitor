#!/usr/bin/env python3
"""ride_longevity.py — early detection of breakouts that will have LONG rides.

The ride rule (enter on 12m momentum, exit on 3m rollover) is a blunt instrument:
it enters most breakouts and exits on the first short-term dip. The literature on
trend durability (Carhart momentum, Daniel & Moskowitz 2016 crash risk, Georges
& Hwang near-high persistence, and vol-scaled momentum) points at a handful of
features that separate breakouts that run from breakouts that fake out:

  1. SMOOTHNESS  — a low-volatility advance persists (vol-scaled momentum beats
                   raw momentum; Griffin, Ji & Martin). A choppy spike is more
                   likely to mean-revert.
  2. PULLBACK RESILIENCE — shallow pullbacks inside the advance (not breaking the
                   recent swing structure) = institutional holding, longer ride.
  3. NON-OVERSHOOT  — price not stretched far above its trend (over-extended
                   moves cluster-revert). Measured vs a trailing momentum MA.
  4. VOLUME ACCUMULATION — sustained institutional buying (OBV trend) rather than
                   a single-volume spike. Confirmed accumulation extends rides.
  5. FUNDAMENTAL SUPPORT — durable earnings (positive ROE / earnings stability)
                   backs a trend that isn't just multiple expansion.

All measures are computed on DAILY data from a price (and optional volume /
fundamentals) series. Returns a per-date DataFrame so each signal can be studied
over time, plus a composite `long_ride_score`.

The composite is used to GATE RIDE EXTENSION (not entry): we only EXTEND a ride
(keep holding past the classic 3m-rollover exit) when the trend is durable, and
we refuse to ENTER a "fresh breakout" that already shows fragility (overshoot +
chop + no accumulation) — those are the fake-outs that round-trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def smoothness(close: pd.Series, window: int = 60) -> pd.Series:
    """Low-vol advance persistence: |mean daily ret| / std over the window.

    A high value = the move is steady (trend returns dominate noise) — the
    signature of a durable ride. A low value = choppy/spikey (vol dominates),
    more likely to mean-revert.
    """
    r = close.pct_change()
    m = r.rolling(window, min_periods=10).mean()
    s = r.rolling(window, min_periods=10).std().replace(0, np.nan)
    return (m / s).rename("smoothness")


def pullback_depth(close: pd.Series, window: int = 60) -> pd.Series:
    """Shallow-pullback resilience: inverse of the worst drawdown in the window.

    Computes the rolling max drawdown from the running peak. A durable advance
    holds its swing structure (small drawdown); a fragile one gives back a lot.
    Returns 1 - |max_dd| (1 = no drawdown, 0 = -100%).
    """
    peak = close.rolling(window, min_periods=10).max()
    dd = close / peak - 1.0
    return (dd.rolling(window, min_periods=10).min() + 1.0).rename("pullback_depth")


def overshoot(close: pd.Series, window: int = 60) -> pd.Series:
    """How far price is stretched above its trailing trend (vol-scaled).

    = (close - rolling_mean) / rolling_std over the window. A large positive
    overshoot = over-extended (cluster-revert risk); near/negative = not
    stretched (room to run). This is the main "avoid the fake-out" filter.
    """
    m = close.rolling(window, min_periods=10).mean()
    s = close.rolling(window, min_periods=10).std().replace(0, np.nan)
    return ((close - m) / s).rename("overshoot")


def volume_accumulation(close: pd.Series, volume: pd.Series,
                        window: int = 60) -> pd.Series:
    """Sustained institutional accumulation: OBV slope over the window.

    OBV rises when up-days out-volume down-days. A steadily rising OBV over the
    advance (positive slope) = accumulation backs the trend. Returns the OBV
    regression slope, rescaled (a 0/1-ish reading via sign + magnitude).
    """
    # ffill volume so NaN rows don't poison the OBV cumsum; align to close index
    v = volume.reindex(close.index).ffill().fillna(0.0)
    sign = np.sign(close.diff().fillna(0))
    obv = (sign * v).cumsum()
    slope = obv.rolling(window, min_periods=10).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    return slope.rename("volume_accumulation")


def durability(close: pd.Series, volume: pd.Series | None = None,
               *, window: int = 60) -> pd.DataFrame:
    """Combine the trend-durability measures into one per-date frame."""
    out = pd.DataFrame(index=close.index)
    out["smoothness"] = smoothness(close, window)
    out["pullback_depth"] = pullback_depth(close, window)
    out["overshoot"] = overshoot(close, window)
    if volume is not None:
        out["volume_accumulation"] = volume_accumulation(close, volume, window)
    else:
        out["volume_accumulation"] = np.nan
    return out


def long_ride_score(close: pd.Series, volume: pd.Series | None = None,
                    *, window: int = 60,
                    fundamentals: dict | None = None) -> pd.DataFrame:
    """Composite early-detection score for a LONG, durable ride.

    Combines the durability measures into a 0-1+ score. A HIGH score at/near a
    breakout = the trend is smooth, holds its pullbacks, isn't over-extended,
    and has accumulation backing it — the signature of a ride that will run.

    Component contributions (each normalized to ~[0,1] where 1 = durable):
      smoothness*0.30         — steady trend (winsorized, scaled)
      pullback_depth*0.20     — shallow pullbacks (already ~[0,1])
      (1-overshoot_norm)*0.20 — not over-extended (overshoot winsorized to [0,1])
      accumulation*0.15       — sustained OBV up-slope (normalized)
      fundamental*0.15        — durable earnings (if provided), else +0.15 neutral

    `fundamentals` dict may carry: roe (0..1+), earnings_stability (0..1).
    Returns a DataFrame with each component + the composite `long_ride_score`.
    """
    d = durability(close, volume, window=window)
    r = close.pct_change().rolling(window, min_periods=10).std()

    # normalize each to ~[0,1]
    smooth_n = np.clip((d["smoothness"] + 1) / 4.0, 0, 1)          # ~[-1,3] -> [0,1]
    pull_n = np.clip(d["pullback_depth"], 0, 1)                    # already [0,1]
    over_n = np.clip((d["overshoot"] + 2) / 4.0, 0, 1)             # ~[-2,2] -> [0,1]
    not_over_n = 1.0 - over_n                                      # high = room to run
    if volume is not None:
        acc_n = np.clip(np.sign(d["volume_accumulation"]) *
                        np.tanh(np.abs(d["volume_accumulation"])), 0, 1)
        acc_n = acc_n.fillna(0.5)   # missing/NaN accumulation = neutral (not evidence of either)
    else:
        acc_n = pd.Series(0.5, index=close.index)
    if fundamentals:
        roe = fundamentals.get("roe")
        estab = fundamentals.get("earnings_stability")
        fund = 0.0
        if roe is not None:
            fund += 0.5 * np.clip(roe, 0, 1)
        if estab is not None:
            fund += 0.5 * np.clip(estab, 0, 1)
    else:
        fund = 0.5

    d["smooth_n"] = smooth_n
    d["pull_n"] = pull_n
    d["not_over_n"] = not_over_n
    d["acc_n"] = acc_n
    d["fund_n"] = pd.Series(fund, index=close.index)
    d["long_ride_score"] = (
        0.30 * smooth_n + 0.20 * pull_n + 0.20 * not_over_n +
        0.15 * acc_n + 0.15 * d["fund_n"]
    )
    return d


def ride_gate(m: pd.Series, *, entry_thresh: float = 0.40,
              stack_depth: int = 0, long_ride: float = 0.0,
              reliability: str = "low") -> dict:
    """Quality-based ride ENTRY gate that does NOT require 12 months of history.

    The classic rule needs 12m momentum (so 12mo of data). This gate opens on
    the LONGEST momentum horizon actually available, and uses SIGNAL QUALITY
    (fractal stack depth + durability score) as the confidence substitute that
    history length used to provide. A name with only 4 months of a clean,
    durable, multi-granular breakout is a better ride than a name with 12 months
    of choppy, over-extended momentum.

    Horizon ladder (pick the longest that has enough data):
      12mo+ -> mom12 > thresh          (full classic signal)
       6mo  -> mom6  > thresh          (intermediate — young_gate reliable)
       3mo  -> mom3  > thresh          (short — needs STRONG quality to open)
    Quality bar:
      >=12mo: stack_depth >= 1 OR long_ride >= 0.35        (light bar)
       6mo : stack_depth >= 2 OR long_ride >= 0.40         (medium bar)
       3mo : stack_depth >= 3 AND long_ride >= 0.45        (strict bar)
    All horizons still require mom1 > 0 (continuation) — the freshest check.

    Returns: gate_open, horizon_used, mom_used (annualized), quality_ok,
             reasons (what blocked it, if closed).
    """
    m = m.replace([np.inf, -np.inf], np.nan).dropna()
    reasons = []
    cum = m.cumsum()
    n = len(m)
    mom1 = (cum.iloc[-1] - cum.iloc[-2]) if n >= 2 else np.nan
    cont = pd.notna(mom1) and mom1 > 0

    if n >= 12:
        horizon = "12mo"
        mom = cum.iloc[-1] - cum.iloc[-12]
        mom_ann = mom
        bar = (stack_depth >= 1) or (long_ride >= 0.35)
    elif n >= 6:
        horizon = "6mo"
        mom = cum.iloc[-1] - cum.iloc[-6]
        mom_ann = mom * 2
        bar = (stack_depth >= 2) or (long_ride >= 0.40)
    elif n >= 3:
        horizon = "3mo"
        mom = cum.iloc[-1] - cum.iloc[-3]
        mom_ann = mom * 4
        bar = (stack_depth >= 3) and (long_ride >= 0.45)
    else:
        horizon = "none"
        mom_ann = np.nan
        bar = False
        reasons.append("insufficient_history")

    mom_ok = pd.notna(mom_ann) and mom_ann > entry_thresh and mom_ann > 0
    gate = bool(mom_ok and cont and bar)

    if not mom_ok:
        reasons.append(f"mom_{horizon}_below_gate")
    if not cont:
        reasons.append("no_1m_continuation")
    if mom_ok and cont and not bar:
        reasons.append("quality_too_low")
    return {
        "gate_open": gate,
        "horizon": horizon,
        "mom_used": round(mom_ann, 4) if pd.notna(mom_ann) else np.nan,
        "stack_depth": int(stack_depth),
        "long_ride": round(float(long_ride), 4),
        "quality_ok": bool(bar),
        "reasons": reasons,
        "reliability": reliability,
    }


def ride_exit(m: pd.Series, *, exit_thresh: float = 0.0,
              stack_depth: int = 0, long_ride: float = 0.0,
              trailing_stop: float | None = None,
              persist: int = 1) -> dict:
    """Dual-condition ride-OVER (exit) test — exits only on CONFIRMED breakdown.

    The classic rule exits when 3m momentum <= 0. That has two failure modes:
      1. PREMATURE EXIT — a brief dip (single negative month) inside a durable
         uptrend triggers exit and you miss the rest of the ride.
      2. LOST GAINS — riding a real breakdown too long because 3m stays positive
         while price gives back the whole move.

    Fix: exit only when the short-term rollover is CONFIRMED by signal quality
    AND (optionally) persists for `persist` consecutive months.

      exit_soft = mom3 <= exit_thresh                  (short-term rollover)
      confirm   = stack_depth <= 1 OR long_ride < 0.35 (trend durability broke)
      hard_stop = price <= trailing_stop (if given, on DAILY close basis)
      EXIT when (exit_soft AND confirm) for `persist` months, OR hard_stop hit.

    `persist=2` means two consecutive months of (rollover AND confirmation)
    before exiting — this is the anti-whipsaw setting: a single weak month in a
    durable ride is held (pullback), but a sustained breakdown exits.

    Rationale (from the durability features): a strong ride can dip a month
    without its multi-granular stack collapsing. If the fractal stack still has
    >=2 views confirmed and the durability score is decent, a negative 3m month
    is a PULLBACK (hold), not an exit. Only when the stack flattens AND momentum
    rolls over persistently do we exit — plus an absolute trailing stop caps the
    loss case.

    m: monthly log returns. stack_depth / long_ride: current fractal-durability.
    trailing_stop: optional max drawdown from running peak (e.g. -0.20).
    persist: consecutive months of rollover+confirm required to exit (>=1).
    Returns: exit (bool), exit_kind (rollover_confirm|trailing_stop|none),
             mom3, confirm, reasons.
    """
    m = m.replace([np.inf, -np.inf], np.nan).dropna()
    reasons = []
    cum = m.cumsum()
    n = len(m)
    mom3 = (cum.iloc[-1] - cum.iloc[-3]) if n >= 3 else np.nan
    exit_soft = pd.notna(mom3) and mom3 <= exit_thresh
    confirm = (stack_depth <= 1) or (long_ride < 0.35)
    hard_stop_hit = False
    if trailing_stop is not None and n >= 1:
        eq = (1 + m).cumprod()
        dd = eq / eq.cummax() - 1.0
        hard_stop_hit = float(dd.iloc[-1]) <= trailing_stop

    if hard_stop_hit:
        exit_flag, kind = True, "trailing_stop"
        reasons.append("hard_stop")
    elif exit_soft and confirm and persist <= 1:
        exit_flag, kind = True, "rollover_confirm"
        reasons.append("3m_rollover")
        reasons.append("stack_or_durability_broke")
    elif exit_soft and not confirm:
        exit_flag, kind = False, "none"
        reasons.append("3m_dip_but_stack_holds")  # pullback, hold
    else:
        # either not soft, or soft+confirm but persist>1 (needs consecutive months)
        exit_flag, kind = False, "none"
        if exit_soft and confirm:
            reasons.append("rollover_not_persistent")

    return {
        "exit": bool(exit_flag),
        "exit_kind": kind,
        "mom3": round(mom3, 4) if pd.notna(mom3) else np.nan,
        "stack_depth": int(stack_depth),
        "long_ride": round(float(long_ride), 4),
        "confirm": bool(confirm),
        "hard_stop": bool(hard_stop_hit),
        "reasons": reasons,
    }
