#!/usr/bin/env python3
"""Kalman gain path analysis for (mkt_ret, log_vol) filter."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
HMM = DATA_DIR / "hmm_regime_states.parquet"
OUT = DATA_DIR / "kalman_gain_path.csv"
OUT_SUM = DATA_DIR / "kalman_gain_summary.csv"

def kf_with_gains(obs, Q_scale=5e-5, R_scale=2e-2):
    T, d = obs.shape
    F, H = np.eye(d), np.eye(d)
    Q, R = np.eye(d) * Q_scale, np.eye(d) * R_scale
    x, P = np.zeros(d), np.eye(d)
    xs = np.zeros((T, d))
    gains = np.zeros((T, d, d))
    innov = np.zeros((T, d))
    innov_std = np.zeros((T, d))
    for t in range(T):
        x = F @ x
        P = F @ P @ F.T + Q
        y = obs[t]
        if np.any(np.isnan(y)):
            xs[t] = x
            continue
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        e = y - H @ x
        x = x + K @ e
        P = (np.eye(d) - K @ H) @ P
        xs[t] = x
        gains[t] = K
        innov[t] = e
        innov_std[t] = np.sqrt(np.diag(S))
    return xs, gains, innov, innov_std

def run(save=True):
    prices = pd.read_parquet(PRICES, columns=["date","ticker","close"])
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    rets = np.log(wide / wide.shift(1)).dropna(how="all")
    mkt = rets.mean(axis=1)
    vol21 = mkt.rolling(21).std() * np.sqrt(252)
    df = pd.DataFrame({"mkt_ret": mkt, "vol21": vol21}).dropna()
    obs = np.column_stack([df.mkt_ret.values, np.log(df.vol21.clip(1e-4).values)])
    xs, gains, innov, innov_std = kf_with_gains(obs)
    out = df.copy()
    out["kf_level"] = xs[:,0]
    out["kf_vol"] = np.exp(xs[:,1])
    out["gain_ret"] = gains[:,0,0]
    out["gain_vol"] = gains[:,1,1]
    out["cross_gain_vol_from_ret"] = gains[:,1,0]
    out["innov_ret"] = innov[:,0]
    out["innov_log_vol"] = innov[:,1]
    out["innov_vol_z"] = innov[:,1] / np.where(innov_std[:,1]==0, np.nan, innov_std[:,1])
    out = out.reset_index().rename(columns={"index":"date"})
    if HMM.exists():
        h = pd.read_csv(HMM)
        h["date"] = pd.to_datetime(h["date"])
        out = out.merge(h[["date","regime"]], on="date", how="left")
    print(out[["gain_ret","gain_vol"]].describe().to_string())
    if "regime" in out.columns:
        print(out.groupby("regime")[["gain_ret","gain_vol","innov_vol_z"]].mean().to_string())
    summary = pd.DataFrame([{"mean_gain_ret": out.gain_ret.mean(), "mean_gain_vol": out.gain_vol.mean(),
        "corr_gain_vol_vs_vol": float(out.gain_vol.corr(out.vol21))}])
    if save:
        out.to_csv(OUT, index=False)
        summary.to_csv(OUT_SUM, index=False)
        print("Wrote", OUT)
    return out

if __name__ == "__main__":
    run(save=True)
