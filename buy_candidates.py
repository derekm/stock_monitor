#!/usr/bin/env python3
"""buy_candidates.py — Decision layer beyond dual-pass gates for names expected to rise.

Combines:
  - Dual-pass / quality / value decisions
  - Momentum score & residual momentum
  - Factor composite
  - Regime posture (stress → fewer / higher bar)
  - Liquidity floor
  - Leverage flags
  - SP500 membership (liquidity / benchmark relevance)

Outputs ranked BUY / ACCUMULATE / WATCH / AVOID with reasons.

Usage:
  python buy_candidates.py --save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
PREF = DATA_DIR / "preferred_metrics.csv"
MOM = DATA_DIR / "momentum_metrics.csv"
FAC = DATA_DIR / "factor_panel.csv"
RISK = DATA_DIR / "risk_metrics_ext.csv"
AGG = DATA_DIR / "signal_aggregator_scores.csv"
HMM = DATA_DIR / "hmm_regime_states.csv"
SP500 = DATA_DIR / "sp500_sleeve.csv"
OUT = DATA_DIR / "buy_candidates.csv"
OUT_TOP = DATA_DIR / "buy_candidates_top.csv"
FRAGILITY = DATA_DIR / "fragility_screen.csv"
SKEW_CSV = DATA_DIR / "options_skew.csv"
SKEW_STEEP = 0.35  # skew >= this (steep put-side fear) triggers the veto

# Taleb veto maps, loaded once in build(): ticker -> fragile flag, ticker -> skew
fragility_map: dict[str, bool] | None = None
skew_map: dict[str, float] | None = None


def regime() -> str:
    if not HMM.exists():
        return "normal"
    h = pd.read_csv(HMM)
    h["date"] = pd.to_datetime(h.get("date"), errors="coerce")
    h = h.dropna(subset=["date"]).sort_values("date")
    if h.empty:
        return "normal"
    for c in ("regime", "state", "label"):
        if c in h.columns:
            return str(h.iloc[-1][c])
    return "normal"


def regime_stress_prob() -> float:
    """Posterior probability of the stress regime (soft stress belief).

    The American-options lesson, fixed: the HMM regime label was used as a
    hard stress verdict (stress → flat -0.08 haircut), and the hidden-
    optionality audit showed that cliff flips 28.4% of decisions when the
    verdict changes. Consuming the posterior p(stress) instead makes the
    haircut continuous in belief: score -= 0.08 * p(stress). A small
    perturbation of the posterior now moves decisions proportionally instead
    of cliff-flipping. Returns p(stress) in [0,1]; 0.0 when unavailable
    (no file / no stress state / degenerate posterior).
    """
    if not HMM.exists():
        return 0.0
    h = pd.read_csv(HMM)
    h["date"] = pd.to_datetime(h.get("date"), errors="coerce")
    h = h.dropna(subset=["date"]).sort_values("date")
    if h.empty:
        return 0.0
    last = h.iloc[-1]
    # find the state whose regime label carries "stress" (e.g. high_vol_stress)
    n_state = int(last.get("state_id", -1))
    for c in h.columns:
        if c.startswith("p_state_"):
            i = int(c.split("_")[-1])
            reg_col = None
            for rc in ("regime", "label"):
                if rc in h.columns:
                    reg_col = rc
                    break
            if reg_col is None:
                return 0.0
            # map state index -> its regime label from any row with that state
            mask = h.get("state_id") == i
            if mask.any():
                lab = str(h.loc[mask, reg_col].iloc[0]).lower()
                if "stress" in lab:
                    p = float(last.get(c, 0.0))
                    return float(np.clip(p, 0.0, 1.0))
    return 0.0


def _load_maps():
    """Taleb veto maps: fragility flag (top-10% pctile) and latest IV skew."""
    fragility_map, skew_map = None, None
    if FRAGILITY.exists():
        fs = pd.read_csv(FRAGILITY)
        fragility_map = dict(zip(fs["ticker"].astype(str).str.upper(), fs["fragile_flag"] == True))
    if SKEW_CSV.exists():
        sk = pd.read_csv(SKEW_CSV)
        sk = sk.sort_values("date") if "date" in sk.columns else sk
        skew_map = dict(zip(sk["ticker"].astype(str).str.upper(), pd.to_numeric(sk["skew"], errors="coerce")))
    return fragility_map, skew_map


def _est_error(series) -> float:
    """Estimation error of a numeric driver: a loose standard error for a
    rolling-window estimate (cross-sectional std / 4). Shared by build() and
    hidden_optionality_audit.py so the production smoothing width matches the
    audit's perturbation scale exactly."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4 or float(s.std()) == 0:
        return 0.0
    return float(s.std()) / 4.0


def _step_eval(x, baseline, steps):
    """Evaluate the exact step function: baseline + Σ Δᵢ·1[x ≥ tᵢ]."""
    v = baseline
    for t, d in steps:
        if x >= t:
            v += d
    return v


def _step_expectation(x, sig, baseline, steps):
    """E[g(x + ε)] for a step function g and ε ~ N(0, sig) — the American-
    options §I-A prescription: integrate the decision over the driver's noise
    distribution instead of evaluating it at the point estimate.

    g is defined by (baseline, steps) with steps = [(threshold, delta)...].
    Its noise-convolved expectation is the closed-form erf blend

        E = baseline + Σ Δᵢ·Φ((x - tᵢ)/sig)

    so the asymptotic credit is EXACTLY preserved while the transitions are
    smoothed over the true noise width. sig <= 0 falls back to the exact
    step function (old behavior)."""
    if sig <= 0:
        return _step_eval(x, baseline, steps)
    import math

    def Phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    return baseline + sum(d * Phi((x - t) / sig) for t, d in steps)


# Numeric decision drivers as step functions (baseline, steps). This is the
# config for _step_expectation — the single source of truth for both the
# exact thresholds (sig=0) and the noise-convolved expectations (sig>0).
MOMENTUM_STEPS = (-0.15, [(-0.5, 0.15), (0.0, 0.10), (0.5, 0.10)])       # -0.15 / 0 / +0.10 / +0.20
FACTOR_STEPS = (0.0, [(0.0, 0.05), (0.5, 0.10)])                        # 0 / +0.05 / +0.15
COMPOSITE_STEPS = (-0.10, [(0.25, 0.10), (0.60, 0.15), (0.75, 0.10)])   # -0.10 / 0 / +0.15 / +0.25
RESID_MOM_STEPS = (0.0, [(0.05, 0.10)])                                 # 0 / +0.10
LIQUIDITY_STEPS = (-0.10, [(0.15, 0.10)])                               # -0.10 / 0
SKEW_STEPS = (0.0, [(0.35, -0.15)])                                     # 0 / -0.15


def score_row(r, stress_p: float, fragility_map=None, skew_map=None, sigs: dict | None = None):
    """Score ONE candidate row -> (score, reasons). Single source of truth for
    the decision loop — shared by build() and hidden_optionality_audit.py so
    the audit perturbs the REAL scorer, not a drifted copy.

    stress_p: soft HMM posterior p(stress) in [0,1] — scales the stress
    haircut continuously (see regime_stress_prob).
    sigs: dict of estimation errors per driver (momentum, factor, composite,
    resid_mom, liquidity, skew). Drivers use their noise-convolved expectation
    when sig > 0; absent keys fall back to the exact thresholds (old
    behavior)."""
    reasons = []
    score = 0.0
    sigs = sigs or {}
    # gate pieces
    dec = r.get("decision")
    if dec == "INCLUDE_CORE":
        score += 0.35; reasons.append("dual_pass_core")
    elif dec == "INCLUDE_VALUE":
        score += 0.20; reasons.append("value_trifecta")
    elif dec == "INCLUDE_QUALITY":
        score += 0.20; reasons.append("buffett_quality")
    elif dec == "SATELLITE":
        score += 0.08; reasons.append("satellite")
    elif dec == "AVOID":
        score -= 0.25; reasons.append("decision_avoid")

    mom = r.get("momentum_score")
    if pd.notna(mom):
        # Momentum contribution = E[g(mom + ε)], ε~N(0, sig): the noise-
        # convolved expectation of the original step function (see
        # _step_expectation). De-noised decision; asymptotic credit identical
        # to the old thresholds.
        m_contrib = _step_expectation(float(mom), sigs.get("momentum", 0.0), *MOMENTUM_STEPS)
        score += m_contrib
        if m_contrib >= 0.15:
            reasons.append("strong_momentum")
        elif m_contrib > 0.0:
            reasons.append("positive_momentum")
        elif m_contrib < 0.0:
            reasons.append("weak_momentum")

    rm = r.get("resid_mom_63")
    if pd.notna(rm):
        score += _step_expectation(float(rm), sigs.get("resid_mom", 0.0), *RESID_MOM_STEPS)
        if rm > 0.05:
            reasons.append("positive_residual_mom")

    fc = r.get("factor_composite")
    if pd.notna(fc):
        f_contrib = _step_expectation(float(fc), sigs.get("factor", 0.0), *FACTOR_STEPS)
        score += f_contrib
        if f_contrib >= 0.10:
            reasons.append("high_factor_composite")

    # signal aggregator: OOS IC-weighted composite (top quintile = strong)
    agg_c = r.get("composite")
    if pd.notna(agg_c):
        a_contrib = _step_expectation(float(agg_c), sigs.get("composite", 0.0), *COMPOSITE_STEPS)
        score += a_contrib
        if a_contrib >= 0.20:
            reasons.append("aggregate_top")
        elif a_contrib >= 0.10:
            reasons.append("aggregate_strong")
        elif a_contrib <= -0.05:
            reasons.append("aggregate_weak")

    if r.get("leverage_flag") == "cheap-assets":
        score += 0.08; reasons.append("cheap_assets_flag")
    elif r.get("leverage_flag") == "levered-assets":
        score -= 0.12; reasons.append("levered_assets_flag")

    liq = r.get("liquidity_score")
    if pd.notna(liq):
        score += _step_expectation(float(liq), sigs.get("liquidity", 0.0), *LIQUIDITY_STEPS)
        if liq < 0.15:
            reasons.append("low_liquidity")

    # Taleb vetoes: fragility (top-10% fragility percentile from the
    # fragility screen) and steep options skew (the market's own fear
    # gauge). Cheap + fragile = skip — fragility is a veto, not a score.
    tk = str(r.get("ticker", "")).upper()
    if fragility_map is not None and fragility_map.get(tk):
        score -= 0.30
        reasons.append("fragile_veto")
    if skew_map is not None and tk in skew_map and pd.notna(skew_map[tk]):
        score += _step_expectation(float(skew_map[tk]), sigs.get("skew", 0.0), *SKEW_STEPS)
        if skew_map[tk] >= SKEW_STEEP:
            reasons.append("skew_steepening")

    if r.get("sp500_member"):
        score += 0.05; reasons.append("sp500_member")

    # Soft stress posture (American-options fix): the HMM posterior
    # p(stress) scales the haircut continuously instead of the old hard
    # verdict (flat -0.08 when "stress" in label). p(stress)~1 behaves
    # like the old stress; p(stress)~0 adds nothing; in between the
    # decision moves proportionally to belief — the audit showed the hard
    # cliff flipped 28.4% of decisions on a 0.10 label perturbation.
    if stress_p > 0.01:
        score -= 0.08 * stress_p
        reasons.append(f"stress_regime_haircut_p{stress_p:.0%}")
        # require stronger momentum in stress for buy (continuous: the penalty
        # scales with the shortfall below the 0.25 anchor, not a cliff at it)
        if pd.isna(mom):
            score -= 0.05 * stress_p
        elif mom < 0.25:
            score -= 0.05 * stress_p * ((0.25 - mom) / 0.25)
    return score, reasons


def action_from_score(score) -> str:
    if score >= 0.55:
        return "BUY"
    if score >= 0.35:
        return "ACCUMULATE"
    if score >= 0.15:
        return "WATCH"
    return "AVOID"


def build() -> pd.DataFrame:
    global fragility_map, skew_map
    fragility_map, skew_map = _load_maps()

    pref = pd.read_csv(PREF) if PREF.exists() else pd.DataFrame()
    if pref.empty:
        return pref
    df = pref.copy()
    for path, cols in (
        (MOM, ["ticker", "momentum_score", "mom_12_1", "ret_21d", "ret_63d", "resid_mom_63", "momentum_quintile"]),
        (FAC, ["ticker", "factor_composite", "f_value", "f_quality", "f_momentum"]),
        (RISK, ["ticker", "adv_dollar_21", "liquidity_score"]),
    ):
        if path.exists():
            extra = pd.read_csv(path)
            keep = [c for c in cols if c in extra.columns]
            if "ticker" in keep:
                df = df.merge(extra[keep], on="ticker", how="left", suffixes=("", "_x"))
    # signal aggregator composite (OOS IC-weighted blend of 5 families)
    if AGG.exists():
        agg = pd.read_csv(AGG)
        keep = [c for c in ("ticker", "composite", "rank") if c in agg.columns]
        if len(keep) >= 2:
            df = df.merge(agg[keep], on="ticker", how="left", suffixes=("", "_agg"))
    if SP500.exists():
        sp = pd.read_csv(SP500)
        df = df.merge(sp[["ticker", "sp500_member", "sp500_sector"]], on="ticker", how="left")
    else:
        df["sp500_member"] = False
        df["sp500_sector"] = df.get("sector")

    reg = regime()
    stress_p_global = regime_stress_prob()
    sigs = {
        "momentum": _est_error(df.get("momentum_score")),
        "factor": _est_error(df.get("factor_composite")),
        "composite": _est_error(df.get("composite")),
        "resid_mom": _est_error(df.get("resid_mom_63")),
        "liquidity": _est_error(df.get("liquidity_score")),
        "skew": _est_error(pd.Series(list(skew_map.values()))) if skew_map else 0.0,
    }
    rows = []
    for _, r in df.iterrows():
        score, reasons = score_row(r, stress_p_global, fragility_map, skew_map, sigs)
        action = action_from_score(score)

        rows.append({
            **{k: r.get(k) for k in (
                "ticker", "sector", "decision", "composite_score", "roe", "roic", "ev_ebitda",
                "pb_ratio", "mktcap_to_assets", "leverage_flag", "momentum_score", "mom_12_1",
                "resid_mom_63", "factor_composite", "liquidity_score", "sp500_member", "sp500_sector",
                "suggested_w_max", "sizing_action", "composite", "rank",
            )},
            "buy_score": round(score, 4),
            "action": action,
            "regime": reg,
            "reasons": ",".join(reasons),
        })
    out = pd.DataFrame(rows).sort_values("buy_score", ascending=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    df = build()
    show = [c for c in ("ticker", "action", "buy_score", "decision", "momentum_score", "factor_composite",
                        "sp500_sector", "reasons") if c in df.columns]
    print(df[show].head(20).to_string(index=False))
    print("\nAction counts:", df["action"].value_counts().to_dict())
    if args.save:
        df.to_csv(OUT, index=False)
        df[df["action"].isin(["BUY", "ACCUMULATE"])].to_csv(OUT_TOP, index=False)
        print(f"Wrote {OUT.name} ({len(df)}), {OUT_TOP.name}")


if __name__ == "__main__":
    main()
