#!/usr/bin/env python3
"""
kalman_state_estimates.py — Kalman filter latent state for market risk.

Local level + stochastic vol proxy:
  State: [latent_return_level, latent_log_vol]
  Observes: mkt_ret, log(vol21)

Also filters average pairwise correlation as a smooth latent corr factor.

Usage:
  python kalman_state_estimates.py --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
HMM = DATA_DIR / "hmm_regime_states.parquet"
OUT = DATA_DIR / "kalman_state_estimates.csv"
OUT_SUM = DATA_DIR / "kalman_state_summary.csv"


def kalman_2d(obs: np.ndarray, Q_scale=1e-4, R_scale=1e-2):
    """Simple linear Gaussian KF for 2D latent, identity observation."""
    T, d = obs.shape
    F = np.eye(d)  # random walk latent
    H = np.eye(d)
    Q = np.eye(d) * Q_scale
    R = np.eye(d) * R_scale
    x = np.zeros(d)
    P = np.eye(d)
    xs = np.zeros((T, d))
    Ps = np.zeros((T, d))
    for t in range(T):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        # update
        y = obs[t]
        if np.any(np.isnan(y)):
            xs[t] = x
            Ps[t] = np.diag(P)
            continue
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (y - H @ x)
        P = (np.eye(d) - K @ H) @ P
        xs[t] = x
        Ps[t] = np.diag(P)
    return xs, Ps


def run(save: bool = True):
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    rets = np.log(wide / wide.shift(1)).dropna(how="all")
    mkt = rets.mean(axis=1)
    vol21 = mkt.rolling(21).std() * np.sqrt(252)

    # avg pairwise corr rolling 21 on subset
    cols = list(rets.columns[:30])
    avg_c = []
    for i in range(len(rets)):
        if i < 21:
            avg_c.append(np.nan)
            continue
        c = rets[cols].iloc[i - 21 : i].corr().values
        mask = np.triu(np.ones(c.shape, bool), 1)
        avg_c.append(float(np.nanmean(c[mask])))
    avg_corr = pd.Series(avg_c, index=rets.index)

    df = pd.DataFrame({"mkt_ret": mkt, "vol21": vol21, "avg_corr": avg_corr}).dropna()
    # observe mkt_ret and log vol
    obs = np.column_stack([
        df["mkt_ret"].values,
        np.log(df["vol21"].clip(lower=1e-4).values),
    ])
    xs, Ps = kalman_2d(obs, Q_scale=5e-5, R_scale=2e-2)
    df = df.copy()
    df["kf_level"] = xs[:, 0]
    df["kf_log_vol"] = xs[:, 1]
    df["kf_vol"] = np.exp(xs[:, 1])
    df["kf_level_var"] = Ps[:, 0]
    df["kf_log_vol_var"] = Ps[:, 1]

    # separate KF on correlation
    obs_c = df["avg_corr"].values.reshape(-1, 1)
    xc, Pc = kalman_2d(
        np.column_stack([obs_c[:, 0], obs_c[:, 0]]),  # dummy 2d
        Q_scale=1e-4, R_scale=5e-3,
    )
    df["kf_corr"] = xc[:, 0]
    df["kf_corr_var"] = Pc[:, 0]

    # compare to HMM if present
    if Path(HMM).exists():
        h = pd.read_csv(HMM)
        h["date"] = pd.to_datetime(h["date"])
        df = df.reset_index().rename(columns={"index": "date"})
        df = df.merge(h[["date", "regime", "vol21"]].rename(columns={"vol21": "hmm_vol"}), on="date", how="left")
    else:
        df = df.reset_index().rename(columns={"index": "date"})

    print("=== Kalman state summary ===")
    print(df[["kf_level", "kf_vol", "kf_corr"]].describe().to_string())
    if "regime" in df.columns:
        print("\n=== KF vol by HMM regime ===")
        print(df.groupby("regime")["kf_vol"].agg(["mean", "std"]).to_string())
        print("\n=== KF corr by HMM regime ===")
        print(df.groupby("regime")["kf_corr"].agg(["mean", "std"]).to_string())

    print("\n=== Last 10 days ===")
    cols_show = ["date", "mkt_ret", "vol21", "kf_vol", "avg_corr", "kf_corr"]
    if "regime" in df.columns:
        cols_show.append("regime")
    print(df[cols_show].tail(10).to_string(index=False))

    summary = pd.DataFrame([{
        "mean_kf_vol": df.kf_vol.mean(),
        "mean_kf_corr": df.kf_corr.mean(),
        "corr_kf_vol_vs_realized": float(df.kf_vol.corr(df.vol21)),
        "corr_kf_corr_vs_realized": float(df.kf_corr.corr(df.avg_corr)),
    }])

    if save:
        df.to_csv(OUT, index=False)
        summary.to_csv(OUT_SUM, index=False)
        print(f"\nWrote {OUT}\nWrote {OUT_SUM}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    run(save=True)


if __name__ == "__main__":
    main()
