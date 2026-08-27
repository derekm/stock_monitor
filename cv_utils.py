#!/usr/bin/env python3
"""
cv_utils.py — Overfitting guards: purged K-fold CV + embargo, Benjamini-Hochberg FDR.

Shared by pair_engine.py, earnings_catalyst.py, cross_section.py so every
reported stat is OOS (train/test temporally disjoint with an embargo gap).

Usage:
  from cv_utils import purged_folds, embargo_mask, bh_fdr, oos_stats_vs_baseline
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def embargo_mask(
    n: int,
    test_start: int,
    test_end: int,
    embargo: int = 21,
) -> np.ndarray:
    """Boolean mask of length n; True for indices that must NOT train on.

    Excludes the test window plus ``embargo`` trading days on either side
    (past information leaks forward through autocorrelation; future leaks
    backward through label lookahead).
    """
    mask = np.zeros(n, dtype=bool)
    lo = max(0, test_start - embargo)
    hi = min(n, test_end + embargo)
    mask[lo:hi] = True
    return mask


def purged_folds(
    n: int,
    n_folds: int = 5,
    embargo: int = 21,
    min_train: int = 252,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sequential purged K-fold: returns [(train_idx, test_idx), ...].

    Test blocks are contiguous and non-overlapping in chronological order;
    each train set excludes the test block +/- embargo. Folds with fewer
    than ``min_train`` train rows are dropped (can't honestly train).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    boundaries = np.linspace(0, n, n_folds + 1).astype(int)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        ts, te = boundaries[k], boundaries[k + 1]
        if te - ts < 20:  # degenerate test block
            continue
        excl = embargo_mask(n, ts, te, embargo)
        train = np.where(~excl)[0]
        test = np.arange(ts, te)
        if len(train) < min_train:
            continue
        folds.append((train, test))
    return folds


def cpcv_folds(
    n: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 21,
    min_train: int = 126,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Combinatorial purged CV (López de Prado): every combo of n_test_groups
    contiguous-date groups is a test set; train is the rest minus embargo."""
    from itertools import combinations

    if n_groups < 3 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("need 1 <= n_test_groups < n_groups")
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [np.arange(edges[i], edges[i + 1]) for i in range(n_groups)]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for combo in combinations(range(n_groups), n_test_groups):
        test = np.concatenate([groups[i] for i in combo])
        excl = np.zeros(n, dtype=bool)
        for i in combo:
            ts, te = int(groups[i][0]), int(groups[i][-1]) + 1
            excl |= embargo_mask(n, ts, te, embargo)
        train = np.where(~excl)[0]
        if len(train) < min_train or len(test) < 20:
            continue
        folds.append((train, test))
    return folds


def bh_fdr(pvals: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg FDR control. Returns boolean array, True = survives.

    Handles NaNs (treated as non-significant) and returns same shape as input.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.zeros(p.shape, dtype=bool)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return out
    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    # BH threshold: largest k where p_(k) <= alpha * k / m
    thresholds = alpha * np.arange(1, m + 1) / m
    sig = ranked <= thresholds
    # monotone step-up: once a rank is significant, all larger ranks are too
    # (take the last True index in order)
    if sig.any():
        last = np.max(np.where(sig)[0])
        sig[: last + 1] = True
    pos = np.where(finite)[0][order][sig]
    out[pos] = True
    return out


def oos_stats_vs_baseline(
    model_rets: pd.Series,
    baseline_rets: pd.Series,
    rf: float = 0.04,
) -> dict:
    """OOS performance of model vs baseline (persistence/equal-weight).

    Both series must be aligned daily returns on the SAME OOS index.
    Returns a dict of summary stats — the only numbers worth quoting.
    """
    df = pd.concat([model_rets.rename("model"), baseline_rets.rename("baseline")], axis=1).dropna()
    if len(df) < 20:
        return {"n_days": int(len(df)), "note": "insufficient OOS overlap"}
    m = df["model"]
    b = df["baseline"]

    def ann(rets: pd.Series) -> tuple[float, float, float]:
        mu = float(rets.mean() * 252)
        sd = float(rets.std() * np.sqrt(252))
        sh = (mu - rf) / sd if sd > 0 else float("nan")
        return mu, sd, sh

    m_ret, m_vol, m_sh = ann(m)
    b_ret, b_vol, b_sh = ann(b)
    excess = m - b
    info_ratio = float(excess.mean() * 252 / (excess.std() * np.sqrt(252))) if excess.std() > 0 else float("nan")
    win = float((m > b).mean())
    # direction accuracy of model vs baseline sign
    dir_acc = float((np.sign(m) == np.sign(b)).mean())
    # extended metric set (Sortino, Calmar) via perf_metrics
    try:
        from perf_metrics import perf_metrics
        pm_m = perf_metrics(m, rf=rf)
        pm_b = perf_metrics(b, rf=rf)
    except Exception:
        pm_m, pm_b = {}, {}
    return {
        "n_days": int(len(df)),
        "model_ann_ret": round(m_ret, 4),
        "model_ann_vol": round(m_vol, 4),
        "model_sharpe": round(m_sh, 4),
        "model_sortino": pm_m.get("sortino"),
        "model_calmar": pm_m.get("calmar"),
        "model_max_dd": pm_m.get("max_dd"),
        "baseline_ann_ret": round(b_ret, 4),
        "baseline_ann_vol": round(b_vol, 4),
        "baseline_sharpe": round(b_sh, 4),
        "baseline_sortino": pm_b.get("sortino"),
        "baseline_calmar": pm_b.get("calmar"),
        "baseline_max_dd": pm_b.get("max_dd"),
        "excess_ann_ret": round(m_ret - b_ret, 4),
        "info_ratio": round(info_ratio, 4),
        "win_rate_vs_baseline": round(win, 4),
        "dir_agreement": round(dir_acc, 4),
    }


if __name__ == "__main__":
    # smoke test
    rng = np.random.default_rng(0)
    p = rng.uniform(size=100)
    p[:8] = rng.uniform(0, 0.005, size=8)
    sig = bh_fdr(p, alpha=0.10)
    print(f"FDR: {sig.sum()} of 100 survive at alpha=0.10")
    folds = purged_folds(2000, n_folds=5, embargo=21)
    print(f"purged folds: {len(folds)} usable")
    for i, (tr, te) in enumerate(folds):
        print(f"  fold {i}: train={len(tr)} test={len(te)} overlap={len(set(tr) & set(te))}")
