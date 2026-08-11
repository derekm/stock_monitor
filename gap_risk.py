#!/usr/bin/env python3
"""gap_risk.py — overnight gap exposure (the Taleb layer).

Why: most catastrophic equity losses happen in OVERNIGHT GAPS, not intraday
drift. Close-to-close backtests cannot see this: a name can look calm while
its open-vs-prev-close gaps carry all the tail risk. This script measures:

1. Gap = open_t / close_{t-1} - 1 per name (we DO store OHLCV).
2. Gap frequency / max gap / gap share of total variance per name.
3. The "gap tail": P(|gap| > 3% ) and P(|gap| > 5%) — the names whose risk
   lives in gaps are the ones that will gap through your stop losses.
4. Worst gap events for the monitored universe (a watchlist for kill switches).

Outputs:
  gap_risk.csv        per-ticker gap stats (mean gap, gap sd, gap share of
                      total variance, P(|gap|>3%), P(|gap|>5%), max gap)
  gap_events.csv      top-N worst single-day gaps with dates

Reads: daily_prices.parquet (date, ticker, open, close).
Usage: python gap_risk.py [--top-events 40]
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-events", type=int, default=40)
    args = ap.parse_args()

    cols = ["date", "ticker", "open", "close"]
    d = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=cols)
    d = d.sort_values(["ticker", "date"])

    rows, events = [], []
    for t, g in d.groupby("ticker"):
        g = g.copy()
        g["prev_close"] = g["close"].shift(1)
        g["gap"] = g["open"] / g["prev_close"] - 1
        g["ret"] = g["close"] / g["prev_close"] - 1
        gap = g["gap"].dropna()
        ret = g["ret"].dropna()
        if len(gap) < 200:
            continue
        gap_sd = gap.std()
        ret_sd = ret.std()
        # gap share of total variance (how much of the risk arrives overnight)
        gap_share = float(gap_sd ** 2 / ret_sd ** 2) if ret_sd > 0 else np.nan
        p3 = float(np.mean(np.abs(gap) > 0.03))
        p5 = float(np.mean(np.abs(gap) > 0.05))
        rows.append({
            "ticker": t, "n_obs": len(gap),
            "mean_gap": round(float(gap.mean()), 5),
            "gap_sd": round(float(gap_sd), 5),
            "ret_sd": round(float(ret_sd), 5),
            "gap_share_of_var": round(gap_share, 3),
            "p_abs_gap_gt_3pct": round(p3, 5),
            "p_abs_gap_gt_5pct": round(p5, 6),
            "max_gap_pct": round(float(gap.abs().max() * 100), 2),
            "min_gap_pct": round(float(gap.min() * 100), 2),
        })
        for _, r in g[g["gap"].abs() > 0.05].iterrows():
            events.append({
                "ticker": t, "date": str(r["date"])[:10],
                "gap_pct": round(float(r["gap"] * 100), 2),
                "close_pct": round(float(r["ret"] * 100), 2),
            })

    df = pd.DataFrame(rows).sort_values("ticker")
    df.to_parquet(DATA_DIR / "gap_risk.parquet")

    ev = (pd.DataFrame(events).sort_values("gap_pct", key=lambda s: s.abs(), ascending=False)
          .head(args.top_events) if events else pd.DataFrame())
    ev.to_parquet(DATA_DIR / "gap_events.parquet")

    print(f"gap_risk.csv: {len(df)} tickers")
    print(f"gap_events.csv: {len(ev)} events")
    if len(df):
        # names whose risk is most concentrated in gaps
        risky = df.sort_values("gap_share_of_var", ascending=False).head(8)
        print("\nHighest gap share of variance — risk arrives overnight, backtests miss it:")
        print(risky[["ticker", "gap_share_of_var", "p_abs_gap_gt_3pct", "max_gap_pct"]].to_string(index=False))
        big = df.sort_values("p_abs_gap_gt_5pct", ascending=False).head(5)
        print("\nMost gap-prone names (P(|gap|>5%)):")
        print(big[["ticker", "p_abs_gap_gt_5pct", "max_gap_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
