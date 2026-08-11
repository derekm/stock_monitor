#!/usr/bin/env python3
"""
hmm_posterior_analysis.py — Explore HMM hidden-state posterior probabilities.

Reads hmm_regime_states.csv (from hmm_regime_detection.py) and reports:
  - Posterior mass by regime over time
  - Entropy / uncertainty (days with mixed beliefs)
  - Soft vs hard labels (argmax vs probability threshold)
  - Transition risk when max posterior is weak

Usage:
  python hmm_posterior_analysis.py --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
HMM = DATA_DIR / "hmm_regime_states.parquet"
OUT = DATA_DIR / "hmm_posterior_analysis.parquet"
OUT_UNC = DATA_DIR / "hmm_uncertain_days.parquet"
OUT_SUM = DATA_DIR / "hmm_posterior_summary.parquet"


def run(save: bool = True, uncertain_entropy: float = 0.7, soft_min: float = 0.7):
    if not HMM.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(DATA_DIR / "hmm_regime_detection.py"), "--save"], check=False)

    h = pd.read_parquet(HMM)
    h["date"] = pd.to_datetime(h["date"])
    pcols = [c for c in h.columns if c.startswith("p_state_")]
    # map state_id -> regime name using mode
    id_to_reg = (
        h.groupby("state_id")["regime"].agg(lambda s: s.mode().iloc[0]).to_dict()
    )
    # rename posteriors to regime names
    for sid, reg in id_to_reg.items():
        col = f"p_state_{sid}"
        if col in h.columns:
            h[f"p_{reg}"] = h[col]

    preg = [c for c in h.columns if c.startswith("p_") and c.replace("p_", "") in set(id_to_reg.values())]
    # entropy
    P = h[pcols].values.astype(float)
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    entropy = -(P * np.log(P)).sum(axis=1) / np.log(P.shape[1])  # normalized 0-1
    h["posterior_entropy"] = entropy
    h["max_posterior"] = P.max(axis=1)
    h["hard_regime"] = h["regime"]
    # soft label: only trust if max_posterior >= soft_min else "uncertain"
    argmax = P.argmax(axis=1)
    h["soft_regime"] = [
        id_to_reg.get(int(a), "unknown") if m >= soft_min else "uncertain"
        for a, m in zip(argmax, h["max_posterior"])
    ]

    print("=== Posterior summary ===")
    print(f"Mean max posterior: {h['max_posterior'].mean():.3f}")
    print(f"Mean normalized entropy: {h['posterior_entropy'].mean():.3f}")
    print(f"Days soft-uncertain (max p < {soft_min}): {(h.soft_regime=='uncertain').sum()}")
    print(f"Days high entropy (> {uncertain_entropy}): {(h.posterior_entropy > uncertain_entropy).sum()}")

    print("\n=== Soft regime distribution ===")
    print(h["soft_regime"].value_counts().to_string())

    # disagreement hard vs soft
    disagree = h[h.soft_regime != h.hard_regime]
    print(f"\nHard vs soft disagreements (incl uncertain): {len(disagree)}")

    unc = h[h.soft_regime == "uncertain"][
        ["date", "hard_regime", "max_posterior", "posterior_entropy", "vol21", "avg_corr", "mkt_ret"]
        + [c for c in h.columns if c.startswith("p_") and not c.startswith("p_state")]
    ]
    print("\n=== Sample uncertain days ===")
    print(unc.tail(10).to_string(index=False))

    # when entropy spikes, what happens to next-day vol
    h["vol_next"] = h["vol21"].shift(-1)
    h["entropy_bin"] = pd.cut(h["posterior_entropy"], bins=[0, 0.3, 0.6, 1.0], labels=["low", "mid", "high"])
    print("\n=== Next-day vol by entropy bin ===")
    print(h.groupby("entropy_bin", observed=True)["vol_next"].agg(["mean", "std", "count"]).to_string())

    summary = pd.DataFrame([{
        "mean_max_posterior": h["max_posterior"].mean(),
        "mean_entropy": h["posterior_entropy"].mean(),
        "n_uncertain_soft": int((h.soft_regime == "uncertain").sum()),
        "n_high_entropy": int((h.posterior_entropy > uncertain_entropy).sum()),
        "pct_confident": float((h.max_posterior >= soft_min).mean()),
        **{f"mean_{c}": h[c].mean() for c in preg},
    }])

    if save:
        h.to_parquet(OUT)
        unc.to_parquet(OUT_UNC)
        summary.to_parquet(OUT_SUM)
        print(f"\nWrote {OUT}\nWrote {OUT_UNC}\nWrote {OUT_SUM}")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=True)


if __name__ == "__main__":
    main()
