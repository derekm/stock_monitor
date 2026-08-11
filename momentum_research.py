#!/usr/bin/env python3
"""momentum_research.py — research-grounded momentum measures + young-ticker gate.

Implements the measures the research audit identified for detecting momentum
explosions (and detecting them EARLY, before 12 months of history):

  JFE 2012  Moskowitz-Ooi-Pedersen time-series momentum (TSMOM): sign of the
            past k-month return, volatility-scaled (1/sigma). Robust at 3/6/12
            mo lookbacks (~80% of the 12-mo Sharpe at 3 mo).
  JT 1993   Jegadeesh-Titman cross-sectional momentum at 3/6/9/12-mo formation
            (ranking long past winners / short past losers).
  RFS 2022  Medhat-Schmeling short-term momentum: 1-mo continuation is strong
            among large, liquid, high-turnover stocks (works with 1 month).
  GH 2004   George-Hwang 52-week-high proximity: nearness to the high predicts
            returns that DON'T reverse (~0.45%/mo). For young stocks the
            '52-week high' IS the listing all-time-high.
  Ritter 91 First-day pop / underpricing is a pricing phenomenon, not momentum:
            DROP the first ~1 month of history before measuring.
  Young-gate
            Graduated 'young-ticker' gate: >=6 mo post-IPO history (strict min
            3, first month dropped), annualized 3/6-mo momentum vs a maturity-
            scaled 40% gate, requires 6-mo>0 AND 1-mo>0 (RFS continuation) AND
            near-listing-high (GH anchor), with volatility + liquidity filters
            (short-term momentum only works in liquid high-turnover names).

All functions are pure (operate on monthly log-return Series or a wide price
frame) and safe on short history. No data I/O — callers (shock_ride, momentum_
analytics) wire them to the price store.

Usage:
  from momentum_research import tsmom, jt_momentum, stmom_1m, gw52_high, young_gate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── defaults ──────────────────────────────────────────────────────────────
ENTRY_THRESH = 0.40          # annualized momentum gate for a ride
MIN_POST_IPO_MONTHS = 6      # young-gate history floor (post first-month drop)
STRICT_MIN_MONTHS = 3        # absolute floor before any signal is usable
FIRST_DROP_MONTHS = 1        # Ritter: drop first month (IPO pop / lockup)
VOL_CAP = 1.0                # max annualized vol before filter (100%)
LIQ_TURNOVER_PCT = 0.30      # require >=30th pctile turnover/liquidity


# ── time-series momentum (JFE 2012) ──────────────────────────────────────
def tsmom_signal(m: pd.Series, lookback_months: int = 3,
                 vol_scaled: bool = True) -> pd.Series:
    """Time-series momentum: sign(+ magnitude) of past k-month return.

    m: monthly log returns (DatetimeIndex). Returns a Series of the same
    index = +1 if past k-mo cumulative return > 0 else 0 (long-only signal).
    vol_scaled: weight by 1/sigma (MOP show this is robust).
    """
    cum = m.cumsum()
    past = cum - cum.shift(lookback_months)
    sig = (past > 0).astype(float)
    if vol_scaled:
        sig = sig / (m.rolling(lookback_months).std().replace(0, np.nan) + 1e-9)
        sig = sig.where(sig.notna(), 0.0)
    return sig


def tsmom_stats(m: pd.Series, lookbacks=(3, 6, 12)) -> dict:
    """Backtest TSMOM at several lookbacks; returns per-lookback Sharpe, and
    the current (latest) signal. m = monthly log returns (no first-month drop)."""
    out = {}
    for k in lookbacks:
        sig = tsmom_signal(m, k)
        strat = (sig.shift(1) * m).dropna()   # position shifts 1 month
        if len(strat) >= 6:
            sharpe = strat.mean() / (strat.std() + 1e-9) * np.sqrt(12)
            out[f"tsmom_{k}mo_sharpe"] = round(float(sharpe), 3)
            out[f"tsmom_{k}mo_sig"] = float(sig.iloc[-1]) if len(sig) else np.nan
            out[f"tsmom_{k}mo_return"] = round(float(strat.sum()), 4)
        else:
            out[f"tsmom_{k}mo_sharpe"] = np.nan
            out[f"tsmom_{k}mo_sig"] = np.nan
            out[f"tsmom_{k}mo_return"] = np.nan
    return out


# ── Jegadeesh-Titman cross-sectional momentum (JT 1993) ──────────────────
def jt_momentum(m: pd.Series, formation: int = 6, skip: int = 1,
                holding: int = 6, lookback_months: int = 6) -> dict:
    """JT 1993-style cross-sectional momentum for a SINGLE series is degenerate
    (no cross-section). This returns the k-month formation cumulative return
    (the ranking variable) and a long-only continuation signal = past k-mo
    return > 0 (skipping the last `skip` month, classic 12-1 style).

    m: monthly log returns. lookback_months = formation window.
    """
    cum = m.cumsum()
    # formation return from (t - lookback) to (t - skip) [skip last month]
    past = cum.shift(skip) - cum.shift(lookback_months + skip)
    ret = float(past.iloc[-1]) if len(past) else np.nan
    sig = int(past.iloc[-1] > 0) if len(past) and pd.notna(past.iloc[-1]) else 0
    return {
        "jt_ret_%d_%d" % (lookback_months, skip): round(ret, 4),
        "jt_sig_%d_%d" % (lookback_months, skip): sig,
    }


# ── short-term momentum (RFS 2022, Medhat-Schmeling) ─────────────────────
def stmom_1m(m: pd.Series) -> dict:
    """1-month continuation. For LIQUID high-turnover names, 1-mo momentum is
    strong and persists ~12 mo; reversal dominates only in illiquid microcaps.
    Signal = past 1-mo return sign + magnitude. Caller applies the liquidity
    filter; this returns the raw 1-mo return."""
    cum = m.cumsum()
    past1 = cum - cum.shift(1)
    ret = float(past1.iloc[-1]) if len(past1) else np.nan
    return {"stmom_1m_ret": round(ret, 4), "stmom_1m_sig": int(ret > 0) if pd.notna(ret) else 0}


# ── 52-week high proximity (George-Hwang 2004) ───────────────────────────
def gw52_high(m: pd.Series, window_months: int = 12) -> dict:
    """Proximity to the 52-week (or listing) high. GH: nearness to high predicts
    returns that DON'T reverse. For young stocks the '52-w high' is the all-time
    high since listing, so this is computable immediately."""
    cum = m.cumsum()
    if len(cum) == 0:
        return {"gw52_high_prox": np.nan, "gw52_high_sig": 0}
    hi = cum.rolling(window_months, min_periods=1).max()
    prox = cum.iloc[-1] / hi.iloc[-1]  # 1.0 = at high, <1 below
    return {
        "gw52_high_prox": round(float(prox), 4),
        "gw52_high_sig": int(prox >= 0.90),  # within 10% of high
    }


# ── volatility + liquidity (for filters) ─────────────────────────────────
def annualized_vol(m: pd.Series, lookback_months: int = 12) -> float:
    """Annualized realized volatility from monthly log returns."""
    s = m.tail(lookback_months)
    return float(s.std() * np.sqrt(12)) if len(s) >= 2 else np.nan


def turnover_pctile(adv: float | None, adv_series: pd.Series) -> float:
    """Liquidity percentile (0..1) of a name's avg dollar volume vs the universe.
    None if no universe provided. Higher = more liquid."""
    if adv is None or adv_series is None or len(adv_series) == 0:
        return np.nan
    r = (adv_series > adv).mean()  # frac of universe with LOWER adv = pctile
    return float(r)


# ── the graduated young-ticker gate ──────────────────────────────────────
def young_gate(m: pd.Series, *, history_months: int, annual_vol: float | None,
               adv: float | None = None, adv_series: pd.Series | None = None,
               entry_thresh: float = ENTRY_THRESH) -> dict:
    """Graduated 'young-ticker' ride gate for names with < 12 mo of history.

    m: monthly log returns, ALREADY with the first ~1 month dropped (Ritter).
    history_months: post-drop months of history available.
    annual_vol: annualized vol (for the vol filter). None = unknown.
    adv / adv_series: liquidity (for the filter). None = unknown.

    Returns a dict with:
      gate_open   (bool) — the young-ticker ride entry is satisfied
      reasons     (list[str]) — which conditions passed/failed
      mom_3m, mom_6m (float) — annualized trailing 3/6-mo momentum
      signal_age_months (int)
      reliability  (str) — 'low' (<3mo), 'building' (3-5mo), 'reliable' (>=6mo)
    """
    reasons = []
    cum = m.cumsum()
    mom3 = cum.iloc[-1] - cum.iloc[-3] if len(cum) >= 3 else np.nan   # log 3-mo
    mom6 = cum.iloc[-1] - cum.iloc[-6] if len(cum) >= 6 else np.nan   # log 6-mo
    mom1 = cum.iloc[-1] - cum.iloc[-1] if len(cum) >= 1 else np.nan   # log 1-mo
    mom1 = (cum.iloc[-1] - cum.iloc[-2]) if len(cum) >= 2 else np.nan
    # annualize (x4 for 3-mo, x2 for 6-mo; guard against short windows)
    mom3_ann = mom3 * 4 if pd.notna(mom3) else np.nan
    mom6_ann = mom6 * 2 if pd.notna(mom6) else np.nan
    # near listing high (George-Hwang anchor)
    gw = gw52_high(m, window_months=max(6, min(12, len(cum))))
    hi_prox = gw["gw52_high_prox"]

    # reliability by history length
    if history_months < STRICT_MIN_MONTHS:
        reliability = "low"
    elif history_months < MIN_POST_IPO_MONTHS:
        reliability = "building"
    else:
        reliability = "reliable"

    # conditions (scaled gate: require 6-mo annualized >= thresh for reliable,
    # or 3-mo annualized >= thresh as an early flag)
    mom_ok = pd.notna(mom6_ann) and mom6_ann >= entry_thresh and mom6_ann > 0
    mom3_ok = pd.notna(mom3_ann) and mom3_ann >= entry_thresh and mom3_ann > 0
    cont_ok = pd.notna(mom1) and mom1 > 0               # RFS 1-mo continuation
    high_ok = pd.notna(hi_prox) and hi_prox >= 0.80     # GH near-high
    vol_ok = annual_vol is None or annual_vol <= VOL_CAP
    liq_ok = adv is None or adv_series is None or turnover_pctile(adv, adv_series) >= LIQ_TURNOVER_PCT

    # gate: reliable history needs 6-mo momentum; building can use 3-mo as a flag
    if reliability == "reliable":
        gate = mom_ok and cont_ok and high_ok and vol_ok and liq_ok
    elif reliability == "building":
        gate = mom3_ok and cont_ok and high_ok and vol_ok and liq_ok
    else:
        gate = False

    if gate:
        reasons.append("young_gate_open")
    if not mom_ok and reliability == "reliable":
        reasons.append("mom6_below_gate")
    if not cont_ok:
        reasons.append("no_1m_continuation")
    if not high_ok:
        reasons.append("off_high")
    if not vol_ok:
        reasons.append("high_vol")
    if not liq_ok:
        reasons.append("illiquid")

    return {
        "gate_open": bool(gate),
        "reasons": reasons,
        "mom_3m_ann": round(mom3_ann, 4) if pd.notna(mom3_ann) else np.nan,
        "mom_6m_ann": round(mom6_ann, 4) if pd.notna(mom6_ann) else np.nan,
        "mom_1m": round(mom1, 4) if pd.notna(mom1) else np.nan,
        "gw_high_prox": round(hi_prox, 4) if pd.notna(hi_prox) else np.nan,
        "signal_age_months": int(history_months),
        "reliability": reliability,
        "annual_vol": round(annual_vol, 3) if pd.notna(annual_vol) else np.nan,
    }


# ── unified research-momentum report for one series ──────────────────────
def research_report(m: pd.Series, *, first_months_to_drop: int = FIRST_DROP_MONTHS,
                    annual_vol: float | None = None, adv: float | None = None,
                    adv_series: pd.Series | None = None,
                    entry_thresh: float = ENTRY_THRESH) -> dict:
    """Full research-momentum report for a monthly log-return series.

    Handles the Ritter first-month drop internally: for very young names
    (listing within ~13 mo), drop the first month before all measures.
    Returns a dict of all measure outputs + the young-gate verdict.
    """
    m = m.replace([np.inf, -np.inf], np.nan).dropna()
    total = len(m)
    # Ritter: drop first month if the name is young (recent listing)
    m_clean = m.iloc[first_months_to_drop:] if total > first_months_to_drop else m
    post_drop = len(m_clean)

    rep = {
        "history_months_total": int(total),
        "history_months_clean": int(post_drop),
        "is_young": bool(total < 13),
    }
    rep.update(tsmom_stats(m_clean, lookbacks=(3, 6, 12)))
    rep.update(jt_momentum(m_clean, lookback_months=6))
    rep.update(stmom_1m(m_clean))
    rep.update(gw52_high(m_clean))
    if annual_vol is not None:
        rep["annual_vol"] = round(annual_vol, 3)
    rep["young_gate"] = young_gate(
        m_clean, history_months=post_drop, annual_vol=annual_vol,
        adv=adv, adv_series=adv_series, entry_thresh=entry_thresh,
    )
    return rep
