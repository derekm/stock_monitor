#!/usr/bin/env python3
"""
rolling_correlation_windows.py — Rolling pairwise & sector correlation windows.

Outputs:
  rolling_corr_avg_timeseries.csv  — avg/median pairwise corr over time
  rolling_sector_corr_windows.csv  — sector EW rolling corr long format
  rolling_corr_stability_by_asset.csv — per-name avg corr to market + stability

Usage:
  python rolling_correlation_windows.py --windows 21,63,126 --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT_TS = DATA_DIR / "rolling_corr_avg_timeseries.csv"
OUT_SEC = DATA_DIR / "rolling_sector_corr_windows.csv"
OUT_STAB = DATA_DIR / "rolling_corr_stability_by_asset.csv"


def avg_pairwise(corr: pd.DataFrame) -> tuple[float, float]:
    v = corr.values
    n = v.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    vals = v[mask]
    return float(np.nanmean(vals)), float(np.nanmedian(vals))


def run(windows=(21, 63, 126), step: int = 5, max_assets: int = 80, save: bool = True):
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "adj_close"])
    prices = prices.rename(columns={"adj_close": "close"})
    prices["date"] = pd.to_datetime(prices["date"])
    stocks = pd.read_parquet(STOCKS, columns=["ticker", "sector"])
    sector_map = stocks.set_index("ticker")["sector"].to_dict()

    counts = prices.groupby("ticker").size().sort_values(ascending=False)
    tickers = counts.index.tolist()[:max_assets]
    wide = (
        prices[prices.ticker.isin(tickers)]
        .pivot_table(index="date", columns="ticker", values="close")
        .sort_index().ffill()
    )
    rets = np.log(wide / wide.shift(1)).dropna(how="all")
    mkt = rets.mean(axis=1)

    ts_rows = []
    for w in windows:
        for i in range(w, len(rets), step):
            block = rets.iloc[i - w : i]
            avg, med = avg_pairwise(block.corr())
            ts_rows.append({
                "date": rets.index[i],
                "window": w,
                "avg_pairwise_corr": avg,
                "median_pairwise_corr": med,
                "mkt_vol": float(block.mean(axis=1).std() * np.sqrt(252)),
            })
    ts = pd.DataFrame(ts_rows)
    print("=== Rolling avg pairwise corr (last 3 per window) ===")
    for w in windows:
        sub = ts[ts.window == w].tail(3)
        print(sub.to_string(index=False))

    # sector rolling
    sec_rows = []
    secs = sorted(set(sector_map.values()))
    for w in windows:
        for i in range(w, len(rets), max(step, w // 3)):
            block = rets.iloc[i - w : i]
            sret = {}
            for sec in secs:
                cols = [t for t in block.columns if sector_map.get(t) == sec]
                if len(cols) >= 2:
                    sret[sec] = block[cols].mean(axis=1)
            if len(sret) < 2:
                continue
            sc = pd.DataFrame(sret).corr()
            avg, med = avg_pairwise(sc)
            sec_rows.append({
                "date": rets.index[i], "window": w,
                "avg_sector_corr": avg, "median_sector_corr": med,
            })
            # store all pairs
            cols = list(sc.columns)
            for a_i, a in enumerate(cols):
                for b in cols[a_i + 1 :]:
                    sec_rows.append({
                        "date": rets.index[i], "window": w,
                        "sector_a": a, "sector_b": b, "corr": float(sc.loc[a, b]),
                    })
    sec_df = pd.DataFrame(sec_rows)

    # per-asset rolling corr to market stability
    stab = []
    w = 63
    for t in rets.columns:
        rc = rets[t].rolling(w).corr(mkt)
        stab.append({
            "ticker": t,
            "mean_corr_to_mkt": float(rc.mean()),
            "std_corr_to_mkt": float(rc.std()),
            "last_corr_to_mkt": float(rc.dropna().iloc[-1]) if rc.dropna().size else np.nan,
            "vol_63d": float(rets[t].iloc[-63:].std() * np.sqrt(252)),
        })
    stab_df = pd.DataFrame(stab).sort_values("std_corr_to_mkt", ascending=False)
    print("\nMost unstable corr-to-market names:")
    print(stab_df.head(8).to_string(index=False))

    if save:
        ts.to_csv(OUT_TS, index=False)
        sec_df.to_csv(OUT_SEC, index=False)
        stab_df.to_csv(OUT_STAB, index=False)
        print(f"Wrote {OUT_TS}, {OUT_SEC}, {OUT_STAB}")
    return ts, sec_df, stab_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="21,63,126")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--max-assets", type=int, default=80)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    windows = tuple(int(x) for x in args.windows.split(","))
    run(windows=windows, step=args.step, max_assets=args.max_assets, save=True)


if __name__ == "__main__":
    main()
