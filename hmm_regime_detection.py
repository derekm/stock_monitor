#!/usr/bin/env python3
"""
hmm_regime_detection.py — Gaussian HMM regimes on market returns + vol.

Features (daily):
  - market equal-weight log return
  - trailing 21d realized vol
  - average pairwise correlation proxy (mean abs cross-corr sample on rolling window)

States interpreted post-hoc by mean return / vol ordering:
  low_vol, normal, high_vol_stress

Usage:
  python hmm_regime_detection.py --n-states 3 --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
OUT_STATES = DATA_DIR / "hmm_regime_states.csv"
OUT_SUM = DATA_DIR / "hmm_regime_summary.csv"
OUT_TRANS = DATA_DIR / "hmm_transition_matrix.csv"


def build_features(rets: pd.DataFrame, corr_window: int = 21) -> pd.DataFrame:
    """Vectorized HMM features: mkt_ret, vol21, avg pairwise corr.

    avg_corr computed via pandas C-level rolling().corr() in chunks
    to avoid large intermediate arrays on wide/long baskets.
    """
    mkt = rets.mean(axis=1)
    vol21 = mkt.rolling(21).std() * np.sqrt(252)
    k = rets.shape[1]
    if k >= 2:
        # Process in chunks to avoid massive intermediate corr matrix
        # (N, k, k) can be huge: 16k × 161 × 161 = 415M floats
        chunk_size = min(2000, len(rets))
        avg_corr = np.full(len(rets), np.nan)
        for start in range(0, len(rets), chunk_size):
            end = min(start + chunk_size, len(rets))
            chunk = rets.iloc[start:end]
            if len(chunk) >= corr_window:
                rc = chunk.rolling(corr_window).corr()
                rc_np = rc.values.reshape(len(chunk), k, k)
                tri = np.triu(np.ones((k, k), dtype=bool), 1)
                avg_corr[start:end] = rc_np[:, tri].mean(axis=1)
        # Fill leading NaNs where window not yet filled
        avg_corr[:corr_window-1] = np.nan
    else:
        avg_corr = np.full(len(rets), np.nan)
    feat = pd.DataFrame(
        {"mkt_ret": mkt, "vol21": vol21, "avg_corr": avg_corr}, index=rets.index
    )
    return feat.dropna()



def fit_hmm(feat: pd.DataFrame, n_states: int = 3, seed: int = 7):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise SystemExit("pip install hmmlearn")

    X = feat[["mkt_ret", "vol21", "avg_corr"]].values
    # standardize for numerical stability
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=seed,
        tol=1e-3,
    )
    model.fit(Xz)
    states = model.predict(Xz)
    post = model.predict_proba(Xz)
    return model, states, post, mu, sd


def label_states(feat: pd.DataFrame, states: np.ndarray) -> dict[int, str]:
    tmp = feat.copy()
    tmp["state"] = states
    g = tmp.groupby("state").agg(mean_ret=("mkt_ret", "mean"), mean_vol=("vol21", "mean"), mean_corr=("avg_corr", "mean"))
    # lowest vol -> low_vol; highest vol -> high_vol_stress; middle -> normal
    order = g.sort_values("mean_vol").index.tolist()
    labels = {}
    names = ["low_vol", "normal", "high_vol_stress"] if len(order) == 3 else [f"state_{i}" for i in range(len(order))]
    if len(order) == 2:
        names = ["low_vol", "high_vol_stress"]
    if len(order) > 3:
        names = ["low_vol"] + [f"mid_{i}" for i in range(1, len(order) - 1)] + ["high_vol_stress"]
    for i, s in enumerate(order):
        labels[int(s)] = names[i] if i < len(names) else f"state_{s}"
    return labels, g


def run(n_states: int = 3, save: bool = True):
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    rets = np.log(wide / wide.shift(1)).dropna(how="all")

    feat = build_features(rets)
    model, states, post, mu, sd = fit_hmm(feat, n_states=n_states)
    labels, g = label_states(feat, states)

    out = feat.copy()
    out["state_id"] = states
    out["regime"] = [labels[int(s)] for s in states]
    for k in range(post.shape[1]):
        out[f"p_state_{k}"] = post[:, k]

    # summary
    rows = []
    for sid, name in labels.items():
        sub = out[out.state_id == sid]
        rows.append({
            "state_id": sid,
            "regime": name,
            "n_days": len(sub),
            "pct_time": len(sub) / len(out),
            "mean_ret_ann": float(sub.mkt_ret.mean() * 252),
            "mean_vol": float(sub.vol21.mean()),
            "mean_avg_corr": float(sub.avg_corr.mean()),
            "median_ret": float(sub.mkt_ret.median()),
        })
    summary = pd.DataFrame(rows).sort_values("mean_vol")
    print("=== HMM regime summary ===")
    print(summary.to_string(index=False))

    # transition matrix with labels
    tm = model.transmat_
    labs = [labels[i] for i in range(n_states)]
    trans = pd.DataFrame(tm, index=labs, columns=labs)
    print("\n=== Transition matrix P(to|from) ===")
    print(trans.round(3).to_string())

    # dwell times
    print("\n=== Regime path (last 30 days) ===")
    print(out[["regime", "mkt_ret", "vol21", "avg_corr"]].tail(30).to_string())

    if save:
        out.reset_index().rename(columns={"index": "date"}).to_csv(OUT_STATES, index=False)
        summary.to_csv(OUT_SUM, index=False)
        trans.to_csv(OUT_TRANS)
        print(f"\nWrote {OUT_STATES}\nWrote {OUT_SUM}\nWrote {OUT_TRANS}")
    return out, summary, trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=3)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(n_states=args.n_states, save=True)


if __name__ == "__main__":
    main()
