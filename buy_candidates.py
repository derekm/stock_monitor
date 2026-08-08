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


def build() -> pd.DataFrame:
    global fragility_map, skew_map
    # Taleb veto maps: fragility flag (top-10% pctile) and latest IV skew.
    fragility_map = None
    skew_map = None
    if FRAGILITY.exists():
        fs = pd.read_csv(FRAGILITY)
        fragility_map = dict(zip(fs["ticker"].astype(str).str.upper(), fs["fragile_flag"] == True))
    if SKEW_CSV.exists():
        sk = pd.read_csv(SKEW_CSV)
        sk = sk.sort_values("date") if "date" in sk.columns else sk
        skew_map = dict(zip(sk["ticker"].astype(str).str.upper(), pd.to_numeric(sk["skew"], errors="coerce")))

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
    stress = "stress" in reg.lower()
    rows = []
    for _, r in df.iterrows():
        reasons = []
        score = 0.0
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
            if mom > 0.5:
                score += 0.20; reasons.append("strong_momentum")
            elif mom > 0.0:
                score += 0.10; reasons.append("positive_momentum")
            elif mom < -0.5:
                score -= 0.15; reasons.append("weak_momentum")

        rm = r.get("resid_mom_63")
        if pd.notna(rm) and rm > 0.05:
            score += 0.10; reasons.append("positive_residual_mom")

        fc = r.get("factor_composite")
        if pd.notna(fc):
            if fc > 0.5:
                score += 0.15; reasons.append("high_factor_composite")
            elif fc > 0.0:
                score += 0.05

        # signal aggregator: OOS IC-weighted composite (top quintile = strong)
        agg_c = r.get("composite")
        if pd.notna(agg_c):
            if agg_c >= 0.75:
                score += 0.25; reasons.append("aggregate_top")
            elif agg_c >= 0.60:
                score += 0.15; reasons.append("aggregate_strong")
            elif agg_c <= 0.25:
                score -= 0.10; reasons.append("aggregate_weak")

        if r.get("leverage_flag") == "cheap-assets":
            score += 0.08; reasons.append("cheap_assets_flag")
        elif r.get("leverage_flag") == "levered-assets":
            score -= 0.12; reasons.append("levered_assets_flag")

        liq = r.get("liquidity_score")
        if pd.notna(liq) and liq < 0.15:
            score -= 0.10; reasons.append("low_liquidity")

        # Taleb vetoes: fragility (top-10% fragility percentile from the
        # fragility screen) and steep options skew (the market's own fear
        # gauge). Cheap + fragile = skip — fragility is a veto, not a score.
        tk = str(r.get("ticker", "")).upper()
        if fragility_map is not None and fragility_map.get(tk):
            score -= 0.30
            reasons.append("fragile_veto")
        if skew_map is not None and tk in skew_map and skew_map[tk] >= SKEW_STEEP:
            score -= 0.15
            reasons.append("skew_steepening")

        if r.get("sp500_member"):
            score += 0.05; reasons.append("sp500_member")

        if stress:
            score -= 0.08
            reasons.append("stress_regime_haircut")
            # require stronger momentum in stress for buy
            if pd.isna(mom) or mom < 0.25:
                score -= 0.05

        # action
        if score >= 0.55:
            action = "BUY"
        elif score >= 0.35:
            action = "ACCUMULATE"
        elif score >= 0.15:
            action = "WATCH"
        else:
            action = "AVOID"

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
