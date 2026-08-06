#!/usr/bin/env python3
"""
threshold_logic.py — Reusable dual-pass / regime-aware threshold logic.

Single source of truth for:
  - BASE dual-pass legs
  - REGIME_THRESHOLDS policy table
  - pass/fail evaluation, failed-leg lists, near-miss distance
  - soft regime selection from HMM posteriors

Usage (library):
  from threshold_logic import evaluate_universe, thresholds_for_regime, select_regime

CLI:
  python threshold_logic.py --regime normal --save
  python threshold_logic.py --from-hmm --save
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
HMM = DATA_DIR / "hmm_regime_states.csv"
OUT = DATA_DIR / "threshold_logic_screen.csv"
OUT_RULES = DATA_DIR / "threshold_logic_rules.json"

from analytics_common import BASE_THRESHOLDS  # canonical dual-pass thresholds

REGIME_THRESHOLDS: dict[str, dict[str, Any]] = {
    "low_vol": {
        "roe_min": 0.15, "roic_min": 0.15, "de_max": 1.0,
        "ev_max": 12.0, "pb_max": 2.0, "mca_max": 0.8,
        "label": "quality_at_fair_price",
        "note": "Calm: fair multiples OK if quality holds.",
    },
    "normal": {
        **BASE_THRESHOLDS,
        "label": "base_dual_pass",
        "note": "Default dual-pass.",
    },
    "high_vol_stress": {
        "roe_min": 0.15, "roic_min": 0.15, "de_max": 0.8,
        "ev_max": 8.0, "pb_max": 1.3, "mca_max": 0.45,
        "label": "defensive_dual_tight",
        "note": "Stress: tighter value/leverage; never looser quality.",
    },
    "uncertain": {
        **BASE_THRESHOLDS,
        "label": "base_dual_pass_uncertain",
        "note": "HMM uncertain: fall back to base dual-pass (no easing).",
    },
}


def thresholds_for_regime(regime: str) -> dict[str, Any]:
    return dict(REGIME_THRESHOLDS.get(regime, REGIME_THRESHOLDS["normal"]))


def select_regime(
    hmm_row: pd.Series | None = None,
    soft_min: float = 0.7,
    default: str = "normal",
) -> str:
    """Pick regime from a single HMM posterior row."""
    if hmm_row is None:
        return default
    # prefer explicit regime + max posterior if available
    pcols = [c for c in hmm_row.index if str(c).startswith("p_state_")]
    if pcols:
        ps = np.array([float(hmm_row[c]) for c in pcols], dtype=float)
        if ps.max() < soft_min:
            return "uncertain"
    return str(hmm_row.get("regime", default))


def select_regime_from_hmm_file(path: Path = HMM, soft_min: float = 0.7) -> str:
    if not path.exists():
        return "normal"
    h = pd.read_csv(path)
    h["date"] = pd.to_datetime(h["date"])
    row = h.sort_values("date").iloc[-1]
    return select_regime(row, soft_min=soft_min)


def legs_pass(row: pd.Series, thr: dict) -> dict[str, bool]:
    return {
        "roe": bool(pd.notna(row.get("roe")) and row["roe"] >= thr["roe_min"]),
        "roic": bool(pd.notna(row.get("roic")) and row["roic"] >= thr["roic_min"]),
        "de": bool(pd.notna(row.get("debt_to_equity")) and row["debt_to_equity"] <= thr["de_max"]),
        "ev": bool(pd.notna(row.get("ev_ebitda")) and row["ev_ebitda"] <= thr["ev_max"]),
        "pb": bool(pd.notna(row.get("pb_ratio")) and row["pb_ratio"] <= thr["pb_max"]),
        "mca": bool(pd.notna(row.get("mktcap_to_assets")) and row["mktcap_to_assets"] <= thr["mca_max"]),
    }


def failed_legs(row: pd.Series, thr: dict) -> list[str]:
    return [k for k, v in legs_pass(row, thr).items() if not v]


def is_dual_pass(row: pd.Series, thr: dict | None = None) -> bool:
    thr = thr or BASE_THRESHOLDS
    return all(legs_pass(row, thr).values())


def distance_to_threshold(row: pd.Series, thr: dict) -> dict[str, float]:
    """Signed gaps: positive = passes with room; negative = shortfall."""
    def g(val, lo=None, hi=None):
        if pd.isna(val):
            return float("nan")
        if lo is not None:
            return float(val - lo)
        return float(hi - val)

    return {
        "roe_gap": g(row.get("roe"), lo=thr["roe_min"]),
        "roic_gap": g(row.get("roic"), lo=thr["roic_min"]),
        "de_gap": g(row.get("debt_to_equity"), hi=thr["de_max"]),
        "ev_gap": g(row.get("ev_ebitda"), hi=thr["ev_max"]),
        "pb_gap": g(row.get("pb_ratio"), hi=thr["pb_max"]),
        "mca_gap": g(row.get("mktcap_to_assets"), hi=thr["mca_max"]),
    }


def evaluate_universe(
    fund: pd.DataFrame,
    regime: str = "normal",
) -> pd.DataFrame:
    thr = thresholds_for_regime(regime)
    rows = []
    for _, x in fund.iterrows():
        legs = legs_pass(x, thr)
        fails = [k for k, v in legs.items() if not v]
        gaps = distance_to_threshold(x, thr)
        rows.append({
            "ticker": x.get("ticker"),
            "regime": regime,
            "label": thr.get("label"),
            "dual_pass": len(fails) == 0,
            "n_failed": len(fails),
            "failed_legs": ",".join(fails),
            **{f"pass_{k}": v for k, v in legs.items()},
            **gaps,
            "roe": x.get("roe"), "roic": x.get("roic"),
            "debt_to_equity": x.get("debt_to_equity"),
            "ev_ebitda": x.get("ev_ebitda"), "pb_ratio": x.get("pb_ratio"),
            "mktcap_to_assets": x.get("mktcap_to_assets"),
        })
    return pd.DataFrame(rows)


def latest_fund(path: Path = FUND) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return df.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default=None)
    ap.add_argument("--from-hmm", action="store_true")
    ap.add_argument("--soft-min", type=float, default=0.7)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    if args.from_hmm or args.regime is None:
        regime = select_regime_from_hmm_file(soft_min=args.soft_min)
    else:
        regime = args.regime

    thr = thresholds_for_regime(regime)
    fund = latest_fund()
    out = evaluate_universe(fund, regime)
    dual = out[out.dual_pass]
    print(f"Regime={regime}  label={thr.get('label')}")
    print(f"Thresholds: { {k: thr[k] for k in BASE_THRESHOLDS} }")
    print(f"Dual-pass n={len(dual)}: {dual.ticker.tolist()}")
    print(f"Near (1 leg): {out[out.n_failed==1].ticker.tolist()}")

    if args.save or True:
        out.to_csv(OUT, index=False)
        Path(OUT_RULES).write_text(json.dumps({
            "active_regime": regime,
            "base": BASE_THRESHOLDS,
            "regime_thresholds": REGIME_THRESHOLDS,
            "soft_min_posterior": args.soft_min,
        }, indent=2))
        print(f"Wrote {OUT}\nWrote {OUT_RULES}")


if __name__ == "__main__":
    main()
