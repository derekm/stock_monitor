#!/usr/bin/env python3
"""
gold_beta.py — measure how a stock tracks gold.

Regresses a ticker's daily returns on a gold proxy's daily returns over several
windows and reports beta, correlation and R-squared, plus a rolling beta series so
the stability of the relationship is visible rather than assumed.

Gold proxies available in daily_prices: GLD, IAU, SGOL (bullion ETFs), GDX/NEM/AEM
(miners). Bullion is the macro variable; a miner-vs-miner correlation mostly measures
sector beta, so bullion is the default.

Beta here is an OLS slope on overlapping daily returns, which answers "how much does
the stock move per 1% move in gold". It is descriptive, not a forecast: a high
R-squared over 2004-2026 says the two moved together historically, not that gold
predicts the stock.

Usage:
    python gold_beta.py --ticker B
    python gold_beta.py --ticker B --proxy GLD --windows 63,252,756
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"


def load_returns(tickers: list[str]) -> pl.DataFrame:
    """Aligned daily log returns, inner-joined on date across all tickers."""
    df = (pl.read_parquet(PRICES, columns=["ticker", "date", "close"])
          .filter(pl.col("ticker").is_in(tickers))
          .filter(pl.col("close") > 0)
          .sort(["ticker", "date"]))
    out = None
    for t in tickers:
        s = (df.filter(pl.col("ticker") == t)
             .select(["date", "close"])
             .unique(subset=["date"], keep="last")
             .sort("date"))
        s = s.with_columns(
            (pl.col("close").log() - pl.col("close").log().shift(1)).alias(t)
        ).drop("close").drop_nulls()
        out = s if out is None else out.join(s, on="date", how="inner")
    return out.sort("date")


def stats(y: np.ndarray, x: np.ndarray) -> dict:
    """OLS y = a + b*x with correlation and R^2."""
    if len(y) < 20:
        return {"n": len(y)}
    b, a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"n": len(y), "beta": float(b), "alpha_daily": float(a),
            "corr": r, "r2": r * r,
            "ann_alpha": float(a) * 252.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--proxy", default="GLD")
    ap.add_argument("--windows", default="63,252,756,2520")
    ap.add_argument("--roll", type=int, default=252)
    args = ap.parse_args()

    t, p = args.ticker.upper(), args.proxy.upper()
    r = load_returns([t, p])
    print(f"{t} vs {p}: {r.height:,} overlapping trading days "
          f"({r['date'].min()} -> {r['date'].max()})")
    print()

    y_all = r[t].to_numpy()
    x_all = r[p].to_numpy()

    print(f"{'window':>10} {'n':>7} {'beta':>8} {'corr':>8} {'R^2':>7} {'ann alpha':>11}")
    for w in [int(v) for v in args.windows.split(",")] + [r.height]:
        w = min(w, r.height)
        s = stats(y_all[-w:], x_all[-w:])
        if "beta" not in s:
            continue
        label = "full" if w == r.height else f"{w}d"
        print(f"{label:>10} {s['n']:>7,} {s['beta']:>8.3f} {s['corr']:>8.3f} "
              f"{s['r2']:>7.3f} {s['ann_alpha']:>10.1%}")

    # rolling beta -- is the relationship stable or does it drift?
    w = args.roll
    if r.height > w + 10:
        betas, dates = [], []
        d = r["date"].to_list()
        for i in range(w, r.height, 21):   # monthly steps
            yy, xx = y_all[i - w:i], x_all[i - w:i]
            if xx.std() > 0:
                betas.append(float(np.polyfit(xx, yy, 1)[0]))
                dates.append(d[i - 1])
        b = np.array(betas)
        print()
        print(f"rolling {w}d beta ({len(b)} monthly observations)")
        print(f"  mean {b.mean():.3f}   std {b.std():.3f}   "
              f"min {b.min():.3f}   max {b.max():.3f}")
        print(f"  share of windows with beta > 0: {(b > 0).mean():.1%}")
        print()
        print("  last 8 windows:")
        for dt, bv in list(zip(dates, b))[-8:]:
            print(f"    {dt}  {bv:6.3f}")

    # how does it compare to the broad market, so gold beta is not just market beta?
    try:
        r2 = load_returns([t, p, "SPY"])
        yy = r2[t].to_numpy()
        xg = r2[p].to_numpy()
        xm = r2["SPY"].to_numpy()
        A = np.column_stack([np.ones_like(xg), xg, xm])
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        print()
        print(f"two-factor on {r2.height:,} days shared with SPY:")
        print(f"  gold beta {coef[1]:.3f}   market beta {coef[2]:.3f}")
        print("  (gold beta controlling for the market isolates the gold link)")
    except Exception as e:                                   # noqa: BLE001
        print(f"\ntwo-factor step unavailable: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
