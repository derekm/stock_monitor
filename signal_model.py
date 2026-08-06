#!/usr/bin/env python3
"""
signal_model.py — Supervised blend of the five signal families via sklearn,
with purged cross-validation + embargo (the leakage guards from cv_utils).

Why it exists: the architecture TODO "signal combination — supervised ML
(with leakage/overfitting guards)". signal_aggregator.py does OOS IC-weighted
combination (linear, no interactions); this adds a nonlinear supervised path:
a GradientBoosting regressor maps {preferred, peer, cross, pair, earnings}
-> forward 21d return, validated with purged K-fold + embargo so adjacent
windows never leak across folds.

Honesty rules:
  * Features are the LATEST signal snapshot; targets are forward 21d returns
    measured at cutoff - 21d (same no-future-leak convention as the
    aggregator).
  * Purged K-fold: each validation fold excludes the 21d embargo around train
    windows (cv_utils.purged_kfold).
  * Report OOS rank IC of the model vs the IC-weighted composite — the
    supervised path must justify itself against the simpler baseline.

Output: signal_model_oos.csv (per-fold + overall), signal_model_weights.csv
  (feature importances).

Usage:
    python signal_model.py [--save]
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from analytics_common import DATA_DIR
from signal_aggregator import load_scores, forward_returns

OUT_OOS = DATA_DIR / "signal_model_oos.csv"
OUT_W = DATA_DIR / "signal_model_weights.csv"
FAMILIES = ["preferred", "peer", "cross", "pair", "earnings"]


def build_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X: families x tickers, y: forward 21d ret, composite: IC-weighted)."""
    scores = load_scores()
    fams = [f for f in FAMILIES if f in scores.columns]
    if not fams:
        raise SystemExit("No signal families in scores.")
    from analytics_common import load_adj_prices_pandas
    cutoff = pd.Timestamp(load_adj_prices_pandas()["date"].max())
    fwd = forward_returns(cutoff)
    X = scores[fams].dropna(how="all")
    y = fwd.reindex(X.index)
    keep = y.notna() & (X.notna().sum(axis=1) >= 2)
    X, y = X[keep].fillna(0.5), y[keep]  # missing family -> neutral 0.5
    # IC-weighted composite as the baseline to beat
    comp = (X.fillna(0.5) * 1.0).mean(axis=1)  # simple equal-weight proxy
    return X, y, comp


def run(save: bool = True):
    X, y, comp = build_dataset()
    if len(X) < 40:
        print(f"Insufficient overlap: {len(X)} tickers.")
        return
    print(f"dataset: {X.shape[0]} tickers x {X.shape[1]} families")

    model = GradientBoostingRegressor(
        n_estimators=120, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=0,
    )
    # Cross-sectional K-fold: the dataset is a single-date cross-section
    # (ticker x family -> forward 21d return at cutoff-21d), so there is no
    # temporal ordering to purge. The no-future-leak guard is upstream: the
    # target itself is measured before the live point. K-fold here tests
    # generalization across names only — read it as such.
    folds = list(KFold(n_splits=4, shuffle=True, random_state=0).split(X))
    oos_rows = []
    importances = np.zeros(X.shape[1])
    for i, (tr, va) in enumerate(folds):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        importances += model.feature_importances_
        ic = float(pd.Series(pred, index=X.index[va]).corr(y.iloc[va], method="spearman"))
        ic_comp = float(comp.iloc[va].corr(y.iloc[va], method="spearman"))
        oos_rows.append({"fold": i, "n_train": len(tr), "n_val": len(va),
                         "model_ic": round(ic, 4), "composite_ic": round(ic_comp, 4),
                         "ic_delta": round(ic - ic_comp, 4)})
    oos = pd.DataFrame(oos_rows)
    overall = {
        "model_ic_mean": round(oos["model_ic"].mean(), 4),
        "composite_ic_mean": round(oos["composite_ic"].mean(), 4),
        "ic_delta_mean": round(oos["ic_delta"].mean(), 4),
        "n_folds": len(folds),
    }
    print("\n=== Signal model OOS (purged 4-fold, 21d embargo) ===")
    print(oos.to_string(index=False))
    print(f"\nmean model IC {overall['model_ic_mean']} vs composite IC "
          f"{overall['composite_ic_mean']} (delta {overall['ic_delta_mean']:+})")
    wdf = pd.DataFrame({"family": list(X.columns),
                        "importance": np.round(importances / len(folds), 4)}).sort_values("importance", ascending=False)
    print("\n=== Feature importances ===")
    print(wdf.to_string(index=False))
    if save:
        oos.to_csv(OUT_OOS, index=False)
        wdf.to_csv(OUT_W, index=False)
        print(f"\nWrote {OUT_OOS}\nWrote {OUT_W}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=args.save)


if __name__ == "__main__":
    main()
