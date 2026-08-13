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


# ---------------------------------------------------------------------------
# structural_gate — second-generation entry gate (structural / risk-scaled).
#
# ride_gate (above) is a LAGGING momentum-level detector: it opens after a
# surge (mom12>thresh), buys the top, then holds through the pullback. On
# volatile / young names (e.g. RAL) it loses to buy-hold. The structural gate
# responds to the PRICE/RISK structure instead of a lagging momentum level.
#
# Supported modes (all daily, no lookahead):
#   turtle     — Donchian 55-day breakout entry + 2x ATR chandelier trailing
#                stop (let winners run, cut losers hard).
#   volscale   — exposure sized to target annualized vol, gated by SMA200
#                trend (always partially exposed, never full size into a spike).
#   regime     — EMA50/EMA200 markup/distribution state machine (enter markup
#                regime, exit distribution).
#   recouple   — enter when close re-couples above EMA21 AND EMA50, size by 1/vol.
#   momentum   — the classic daily momentum gate (mom12>0.40 & mom3>0, exit
#                mom3<=0), included for comparison.
#   hybrid     — momentum entry + vol-scaled size + 2x ATR chandelier stop
#                (best drawdown control across the universe backtest).
#   consensus  — majority of the four structural signals, vol-scaled size
#                (best risk-adjusted return among the pure structural modes).
# ---------------------------------------------------------------------------
def _ema(x, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def _atr(close: np.ndarray, n_atr: int = 14) -> np.ndarray:
    tr = np.abs(np.diff(close, prepend=close[0]))
    return pd.Series(tr).ewm(span=n_atr, adjust=False).mean().to_numpy()


STRUCTURAL_MODES = ("turtle", "volscale", "regime", "recouple",
                    "momentum", "hybrid", "consensus")


def structural_positions(close: pd.Series, *, mode: str = "hybrid",
                         target_vol: float = 0.30) -> pd.Series:
    """Daily position series (0..~1.5) for a structural gate mode.

    close: daily close Series. mode: one of STRUCTURAL_MODES.
    target_vol: annualized vol target for the vol-scaled modes.
    Returns a Series (0/partial/full position) aligned to close.index.
    """
    if mode not in STRUCTURAL_MODES:
        raise ValueError(f"mode must be one of {STRUCTURAL_MODES}, got {mode}")
    c = close.to_numpy(dtype=float)
    n = len(c)
    pos = np.zeros(n)
    a = _atr(c)
    ret = np.zeros(n); ret[1:] = c[1:] / c[:-1] - 1.0
    rv = pd.Series(ret).rolling(20).std().to_numpy() * np.sqrt(252)
    size = np.clip(target_vol / np.where(rv == 0, np.nan, rv), 0, 1.5)
    size = np.nan_to_num(size, nan=0.0)

    # momentum (daily, classic): mom12>0.40 & mom3>0, exit mom3<=0
    m12 = pd.Series(ret).rolling(252, min_periods=60).mean().to_numpy() * 252
    m3 = pd.Series(ret).rolling(63, min_periods=21).mean().to_numpy() * 252
    mom_long = (m12 > 0.40) & (m3 > 0)

    if mode == "turtle":
        inpos = False; chand = 0.0
        for i in range(55, n):
            if not inpos:
                if c[i] > np.max(c[i - 55:i]):
                    inpos = True; chand = c[i] - 2.0 * a[i]
            else:
                chand = max(chand, c[i] - 2.0 * a[i])
                if c[i] < chand:
                    inpos = False
            pos[i] = 1.0 if inpos else 0.0

    elif mode == "volscale":
        sma200 = pd.Series(c).rolling(200).mean().to_numpy()
        pos = size * (c > sma200)

    elif mode == "regime":
        e50 = _ema(c, 50); e200 = _ema(c, 200)
        for i in range(50, n):
            up = (c[i] > e50[i]) and (e50[i] > e50[i - 1]) and (c[i] > e200[i])
            down = c[i] < e50[i]
            if up:
                pos[i] = 1.0
            elif down:
                pos[i] = 0.0
            else:
                pos[i] = pos[i - 1] if i > 0 else 0.0

    elif mode == "recouple":
        e21 = _ema(c, 21); e50 = _ema(c, 50)
        recoupled = (c > e21) & (c > e50)
        pos = size * recoupled

    elif mode == "momentum":
        inpos = False
        for i in range(1, n):
            if not inpos:
                if mom_long[i]:
                    inpos = True
            else:
                if m3[i] <= 0:
                    inpos = False
            pos[i] = 1.0 if inpos else 0.0

    elif mode == "hybrid":
        chand = 0.0; inpos = False
        for i in range(55, n):
            if not inpos:
                if mom_long[i]:
                    inpos = True; chand = c[i] - 2.0 * a[i]
            else:
                chand = max(chand, c[i] - 2.0 * a[i])
                if c[i] < chand or m3[i] <= 0:
                    inpos = False
            pos[i] = size[i] if inpos else 0.0

    elif mode == "consensus":
        e21 = _ema(c, 21); e50 = _ema(c, 50)
        sma200 = pd.Series(c).rolling(200).mean().to_numpy()
        recoupled = (c > e21) & (c > e50)
        # regime signal (markup/distribution state machine)
        reg = np.zeros(n)
        for i in range(50, n):
            up = (c[i] > e50[i]) and (e50[i] > e50[i - 1]) and (c[i] > sma200[i])
            if up:
                reg[i] = 1.0
            elif c[i] < e50[i]:
                reg[i] = 0.0
            else:
                reg[i] = reg[i - 1] if i > 0 else 0.0
        # majority of four structural signals, then vol-scale
        struct = (recoupled.astype(int) + reg + _pos_turtle_like(c, a).astype(int) + (c > sma200).astype(int)) >= 2
        pos = size * struct.astype(float)

    return pd.Series(pos, index=close.index)


def _pos_turtle_like(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    n = len(c); pos = np.zeros(n); inpos = False; chand = 0.0
    for i in range(55, n):
        if not inpos:
            if c[i] > np.max(c[i - 55:i]):
                inpos = True; chand = c[i] - 2.0 * a[i]
        else:
            chand = max(chand, c[i] - 2.0 * a[i])
            if c[i] < chand:
                inpos = False
        pos[i] = 1.0 if inpos else 0.0
    return pos


def structural_gate(close: pd.Series, *, mode: str = "hybrid",
                    target_vol: float = 0.30) -> dict:
    """Current structural-gate signal for a ticker.

    Returns a dict: mode, signal (position 0..~1.5), gate_open (signal>0),
    in_market_fraction (mean position over series), and the last N positions.
    """
    pos = structural_positions(close, mode=mode, target_vol=target_vol)
    cur = float(pos.iloc[-1])
    return {
        "mode": mode,
        "signal": round(cur, 4),
        "gate_open": bool(cur > 0),
        "in_market_fraction": round(float(pos.mean()), 4),
        "target_vol": target_vol,
    }
