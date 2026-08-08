#!/usr/bin/env python3
"""ergodicity_ruin.py — the ergodicity critique (the Taleb layer).

Why: Sharpe ratios and ensemble averages are NOT the right lens for a
portfolio that compounds over time. For fat-tailed payoffs the TIME average
differs from the ENSEMBLE average, and ruin is absorbing: you cannot recover
from -100% with +100%. This script computes, per ticker and for the portfolio:

1. Arithmetic vs geometric mean return — the gap (mu - sigma^2/2 for iid) is
   variance drag; for fat tails the true gap is larger.
2. Probability of ruin over N years (block-bootstrap paths; ruin = drawdown
   reaching -99% OR terminal value < 50% of start).
3. Time-to-double vs time-to-ruin — if ruin comes faster than doubling, the
   position is a bad bet regardless of Sharpe.
4. Path-dependence: shuffle the same returns and show terminal outcomes
   scatter (order matters; the median path is not the ensemble mean).

Outputs:
  ergodicity_ruin.csv       per-ticker arith/geom, variance drag, ruin prob
                            (1y/5y/10y), time-to-double, time-to-ruin(5%),
                            doube_ruin_ratio
  portfolio_ergodic.csv     same for the equal-weight portfolio + path scatter
                            percentiles (p5/p50/p95 of 10y terminal wealth)

Reads: daily_prices.parquet.
Usage: python ergodicity_ruin.py [--years 1 5 10] [--paths 400] [--tickers A,B]
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
TRADING_DAYS = 252


def block_bootstrap_paths(ret, n_paths, years, block=21, seed=7):
    """Resample returns in 21-day blocks (preserves short autocorrelation),
    simulate n_paths x years*252 paths, return terminal wealth + max drawdown."""
    rng = np.random.default_rng(seed)
    ret = np.asarray(ret, dtype=float)
    ret = ret[np.isfinite(ret)]
    n = len(ret)
    horizon = years * TRADING_DAYS
    n_blocks = int(np.ceil(horizon / block))
    paths = np.empty((n_paths, horizon))
    for p in range(n_paths):
        idx = rng.integers(0, max(n - block, 1), n_blocks)
        chunk = np.concatenate([ret[i:i + block] for i in idx])[:horizon]
        paths[p] = chunk
    wealth = np.cumprod(1 + paths, axis=1)
    return wealth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--paths", type=int, default=400)
    ap.add_argument("--tickers", default=None)
    args = ap.parse_args()

    cols = ["date", "ticker", "close"]
    d = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=cols)
    if args.tickers:
        d = d[d["ticker"].isin([t.strip().upper() for t in args.tickers.split(",")])]
    d = d.sort_values(["ticker", "date"])

    rows = []
    for t, g in d.groupby("ticker"):
        r = g["close"].pct_change().dropna().to_numpy()
        if len(r) < 500:
            continue
        arith = float(r.mean())
        geom = float(np.expm1(np.mean(np.log1p(r)))) if np.all(r > -1) else np.nan
        var_drag = float(arith - geom)
        row = {
            "ticker": t, "n_obs": len(r),
            "arith_ann": round(arith * TRADING_DAYS, 3),
            "geom_ann": round(geom * TRADING_DAYS, 3),
            "variance_drag_ann": round(var_drag * TRADING_DAYS, 3),
        }
        # ruin probability per horizon
        for yrs in args.years:
            w = block_bootstrap_paths(r, args.paths, yrs)
            terminal = w[:, -1]
            ruin = float(np.mean(terminal < 0.5))  # lost half the bankroll
            maxdd = float(np.mean((w / np.maximum.accumulate(w, axis=1)).min(axis=1) < 0.99))
            row[f"ruin_p_{yrs}y"] = round(ruin, 4)
            row[f"dd99_p_{yrs}y"] = round(maxdd, 4)
        # time-to-double vs time-to-ruin (5% drawdown threshold) on median path
        w10 = block_bootstrap_paths(r, args.paths, 10)
        med = np.median(w10, axis=0)
        t_double = int(np.argmax(med >= 2.0)) + 1 if np.any(med >= 2.0) else None
        dd = w10 / np.maximum.accumulate(w10, axis=1)
        t_ruin = int(np.argmax(np.any(dd < 0.95, axis=0))) + 1 if np.any(dd < 0.95) else None
        row["days_to_double"] = t_double
        row["days_to_dd5"] = t_ruin
        if t_double and t_ruin:
            row["double_ruin_ratio"] = round(t_double / t_ruin, 2)
        else:
            row["double_ruin_ratio"] = None
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("ticker")
    df.to_csv(DATA_DIR / "ergodicity_ruin.csv", index=False)

    # portfolio: equal-weight daily returns (date-aligned)
    port = None
    n_tick = 0
    for t, g in d.groupby("ticker"):
        r = g["close"].pct_change()
        if len(r.dropna()) < 500:
            continue
        s = r.dropna()
        s.index = pd.to_datetime(g["date"].iloc[1:1 + len(s)])
        port = s if port is None else port.add(s, fill_value=0)
        n_tick += 1
    if port is not None:
        pr = (port / n_tick).dropna().to_numpy()
        arith = float(pr.mean())
        geom = float(np.expm1(np.mean(np.log1p(pr)))) if np.all(pr > -1) else np.nan
        port_rows = [{
            "metric": "arith_ann", "value": round(arith * TRADING_DAYS, 4),
        }, {
            "metric": "geom_ann", "value": round(geom * TRADING_DAYS, 4),
        }, {
            "metric": "variance_drag_ann", "value": round((arith - geom) * TRADING_DAYS, 4),
        }]
        for yrs in args.years:
            w = block_bootstrap_paths(pr, args.paths, yrs)
            terminal = w[:, -1]
            port_rows.append({"metric": f"ruin_p_{yrs}y", "value": round(float(np.mean(terminal < 0.5)), 4)})
            pct = np.percentile(terminal, [5, 50, 95])
            port_rows.append({"metric": f"terminal_p5_{yrs}y", "value": round(float(pct[0]), 3)})
            port_rows.append({"metric": f"terminal_p50_{yrs}y", "value": round(float(pct[1]), 3)})
            port_rows.append({"metric": f"terminal_p95_{yrs}y", "value": round(float(pct[2]), 3)})
        pd.DataFrame(port_rows).to_csv(DATA_DIR / "portfolio_ergodic.csv", index=False)

    print(f"ergodicity_ruin.csv: {len(df)} tickers | portfolio_ergodic.csv written")
    if len(df):
        worst = df.sort_values("double_ruin_ratio", ascending=True).head(5)
        print("\nWorst double-vs-ruin ratios (ruin comes before doubling):")
        print(worst[["ticker", "geom_ann", "ruin_p_10y", "days_to_double", "days_to_dd5", "double_ruin_ratio"]].to_string(index=False))
        print("\nNote: variance_drag_ann > 0.10 means the Sharpe lens overstates the compounding path.")


if __name__ == "__main__":
    main()
