#!/usr/bin/env python3
"""research_hygiene.py — Walk-forward inclusion rules + forecast reliability report.

  python research_hygiene.py walk-forward --save
  python research_hygiene.py forecast-reliability --save
  python research_hygiene.py all --save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
PREF_HIST = DATA_DIR / "preferred_metrics_history.parquet"
PREF = DATA_DIR / "preferred_metrics.csv"
FC_BT = DATA_DIR / "forecast_backtest_metrics.csv"
OUT_WF = DATA_DIR / "inclusion_walkforward.csv"
OUT_FR = DATA_DIR / "forecast_reliability_report.csv"


def walk_forward() -> pd.DataFrame:
    """If history exists, measure dual-pass stability across as_of dates; else single snapshot."""
    if PREF_HIST.exists():
        h = pd.read_parquet(PREF_HIST)
    elif PREF.exists():
        h = pd.read_csv(PREF)
        h["as_of_date"] = pd.Timestamp.today().date().isoformat()
    else:
        return pd.DataFrame()

    if "as_of_date" not in h.columns:
        h["as_of_date"] = "snapshot"
    # dual definition
    if "decision" in h.columns:
        h["is_dual"] = h["decision"].eq("INCLUDE_CORE")
    elif {"buffett_pass", "trifecta_pass"}.issubset(h.columns):
        h["is_dual"] = h["buffett_pass"].astype(bool) & h["trifecta_pass"].astype(bool)
    else:
        h["is_dual"] = False

    rows = []
    for dt, g in h.groupby("as_of_date"):
        rows.append({
            "as_of_date": dt,
            "n": len(g),
            "n_dual": int(g["is_dual"].sum()),
            "dual_rate": float(g["is_dual"].mean()),
            "median_composite": float(g["composite_score"].median()) if "composite_score" in g.columns else np.nan,
            "mean_composite": float(g["composite_score"].mean()) if "composite_score" in g.columns else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("as_of_date")
    if len(out) >= 2 and "n_dual" in out.columns:
        out["delta_n_dual"] = out["n_dual"].diff()
    return out


def forecast_reliability() -> pd.DataFrame:
    if not FC_BT.exists():
        return pd.DataFrame()
    df = pd.read_csv(FC_BT)
    # rank by directional accuracy then mae
    if "directional_accuracy" in df.columns:
        df = df.sort_values(["directional_accuracy", "mae"] if "mae" in df.columns else ["directional_accuracy"],
                            ascending=[False, True] if "mae" in df.columns else [False])
    df["reliability_tier"] = pd.cut(
        df["directional_accuracy"] if "directional_accuracy" in df.columns else pd.Series([0]*len(df)),
        bins=[-0.01, 0.45, 0.55, 1.01],
        labels=["weak", "mixed", "strong"],
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("walk-forward", "forecast-reliability", "all"):
        p = sub.add_parser(c)
        p.add_argument("--save", action="store_true")
    args = ap.parse_args()
    if args.cmd in ("walk-forward", "all"):
        wf = walk_forward()
        print("Walk-forward inclusion:")
        print(wf.tail(10).to_string(index=False) if len(wf) else "(empty)")
        if args.save and len(wf):
            wf.to_csv(OUT_WF, index=False)
            print(f"Wrote {OUT_WF}")
    if args.cmd in ("forecast-reliability", "all"):
        fr = forecast_reliability()
        print("Forecast reliability:")
        cols = [c for c in ("ticker", "horizon", "directional_accuracy", "mae", "reliability_tier", "backend") if c in fr.columns]
        print(fr[cols].head(12).to_string(index=False) if len(fr) else "(empty)")
        if args.save and len(fr):
            fr.to_csv(OUT_FR, index=False)
            print(f"Wrote {OUT_FR}")


if __name__ == "__main__":
    main()
