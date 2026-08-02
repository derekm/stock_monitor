#!/usr/bin/env python3
"""black_litterman_views.py — Build BL views from dual-pass / regime posture.

Views:
  - INCLUDE_CORE names: bullish excess return view
  - high_vol_stress: shrink views toward 0 (less conviction)
  - levered-assets flag: dampen view

Usage:
  python black_litterman_views.py --save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
PREF = DATA_DIR / "preferred_metrics.csv"
HMM = DATA_DIR / "hmm_regimes.csv"
OUT = DATA_DIR / "black_litterman_views.csv"
OUT_W = DATA_DIR / "black_litterman_weights_from_views.csv"


def current_regime() -> str:
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


def build_views() -> pd.DataFrame:
    if not PREF.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREF)
    regime = current_regime()
    stress = "stress" in regime.lower()
    rows = []
    for _, r in df.iterrows():
        view = 0.0
        conf = 0.0
        if r.get("decision") == "INCLUDE_CORE" or (r.get("buffett_pass") and r.get("trifecta_pass")):
            view = 0.04  # +4% excess annual view
            conf = 0.6
        elif r.get("decision") == "INCLUDE_VALUE":
            view = 0.02
            conf = 0.4
        elif r.get("decision") == "INCLUDE_QUALITY":
            view = 0.025
            conf = 0.45
        elif r.get("decision") in ("AVOID",):
            view = -0.02
            conf = 0.3
        if r.get("leverage_flag") == "levered-assets":
            view *= 0.5
            conf *= 0.7
        if stress:
            view *= 0.4
            conf *= 0.5
        if view == 0 and conf == 0:
            continue
        rows.append({
            "ticker": r["ticker"],
            "view_excess_ret": round(view, 4),
            "confidence": round(conf, 3),
            "regime": regime,
            "decision": r.get("decision"),
            "leverage_flag": r.get("leverage_flag"),
            "composite_score": r.get("composite_score"),
        })
    return pd.DataFrame(rows).sort_values("view_excess_ret", ascending=False)


def naive_weights(views: pd.DataFrame) -> pd.DataFrame:
    """Confidence-weighted long-only normalized weights on positive views."""
    v = views[views["view_excess_ret"] > 0].copy()
    if v.empty:
        return v
    v["raw"] = v["view_excess_ret"] * v["confidence"]
    v["weight"] = v["raw"] / v["raw"].sum()
    return v[["ticker", "weight", "view_excess_ret", "confidence", "decision"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    views = build_views()
    print(views.head(12).to_string(index=False))
    w = naive_weights(views)
    print("\nWeights:")
    print(w.head(10).to_string(index=False))
    if args.save:
        views.to_csv(OUT, index=False)
        w.to_csv(OUT_W, index=False)
        print(f"Wrote {OUT.name}, {OUT_W.name}")


if __name__ == "__main__":
    main()
