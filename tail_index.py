#!/usr/bin/env python3
"""tail_index.py — fat-tail diagnostics (the Taleb layer).

Why: the Gaussian assumption understates tail risk by orders of magnitude.
This script measures the ACTUAL tail behavior of the price history:

1. Hill estimator tail index alpha per asset + portfolio — alpha < 3 means
   variance is nearly meaningless as a risk measure.
2. Empirical-vs-Gaussian tail probability comparison: P(|ret| > k*sigma) as
   observed vs as a normal would imply. The ratio is the "how wrong is your
   risk model" number.
3. Upper/lower tail dependence between assets (crisis-period co-movement),
   which correlation misses entirely (correlation -> 0 mid-distribution,
   -> 1 in the tails).

Outputs:
  tail_index.csv          per-ticker alpha, empirical P(|z|>5), gaussian P,
                          tail ratio, n_obs
  portfolio_tail.csv      same metrics for the equal-weight portfolio
  tail_dependence.csv     pairwise upper/lower tail dependence (top-N pairs)

Reads: daily_prices.parquet (date, ticker, close).
Usage: python tail_index.py [--tickers A,B,C] [--top-pairs 30]
"""
import argparse
import json
import numpy as np
import pandas as pd

DATA_DIR = __import__("pathlib").Path(__file__).resolve().parent


def load_close(tickers=None):
    cols = ["date", "ticker", "close"]
    d = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=cols)
    if tickers:
        d = d[d["ticker"].isin(tickers)]
    d = d.sort_values(["ticker", "date"])
    return d


def returns_by_ticker(d):
    out = {}
    for t, g in d.groupby("ticker"):
        r = g["close"].pct_change().dropna().to_numpy()
        if len(r) > 300:
            out[t] = r
    return out


def hill_alpha(x, k_frac=0.10):
    """Hill estimator for the tail index of the RIGHT tail (positive extremes).
    Uses the largest k = k_frac*n order statistics. alpha = 1/mean(log(x/x_k))."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    n = len(x)
    k = max(5, int(k_frac * n))
    k = min(k, n - 1)
    xs = np.sort(x)[::-1][:k]
    return float(k / np.sum(np.log(xs / xs[-1]))) if xs[-1] > 0 else np.nan


def tail_probs(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    mu, sd = r.mean(), r.std()
    if sd <= 0:
        return None
    z = (r - mu) / sd
    for k in (3, 4, 5):
        emp = float(np.mean(np.abs(z) > k))
        # gaussian two-sided tail
        from math import erfc
        gauss = float(erfc(k / np.sqrt(2)))
        yield k, emp, gauss, (emp / gauss if gauss > 0 else np.nan)


def tail_dependence(x, y, q=0.05):
    """Upper/lower tail dependence: P(both in the q-quantile tail) / q.
    = 1 means perfect tail co-movement; ~q means independent tails.
    x and y must be DATE-ALIGNED series (same index)."""
    x = pd.Series(x)
    y = pd.Series(y)
    idx = x.index.intersection(y.index)
    if len(idx) < 200:
        return None, None
    x, y = x.loc[idx].to_numpy(), y.loc[idx].to_numpy()
    xu, yu = np.quantile(x, 1 - q), np.quantile(y, 1 - q)
    xl, yl = np.quantile(x, q), np.quantile(y, q)
    up = float(np.mean((x > xu) & (y > yu)) / q)
    lo = float(np.mean((x < xl) & (y < yl)) / q)
    return up, lo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=None, help="comma list; default = all monitored")
    ap.add_argument("--top-pairs", type=int, default=30)
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    d = load_close(tickers)
    rets = returns_by_ticker(d)
    if not rets:
        print("no data")
        return

    rows = []
    for t, r in rets.items():
        alpha = hill_alpha(np.abs(r))  # absolute tails (two-sided extremes)
        probs = {k: (emp, gauss, ratio) for k, emp, gauss, ratio in tail_probs(r)}
        row = {
            "ticker": t, "n_obs": len(r),
            "tail_alpha_hill": round(alpha, 2) if np.isfinite(alpha) else None,
            "emp_p_gt_3sd": round(probs.get(3, (None,))[0], 5) if 3 in probs else None,
            "gauss_p_gt_3sd": round(probs.get(3, (None,))[1], 5) if 3 in probs else None,
            "tail_ratio_3sd": round(probs.get(3, (None,))[2], 1) if 3 in probs else None,
            "emp_p_gt_5sd": round(probs.get(5, (None,))[0], 6) if 5 in probs else None,
            "gauss_p_gt_5sd": round(probs.get(5, (None,))[1], 6) if 5 in probs else None,
            "tail_ratio_5sd": round(probs.get(5, (None,))[2], 1) if 5 in probs else None,
            "kurtosis": round(float(pd.Series(r).kurtosis()), 1),
        }
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("ticker")
    df.to_csv(DATA_DIR / "tail_index.csv", index=False)

    # portfolio-level: equal-weight average of standardized returns (aligned on REAL dates)
    port_srs = None
    n_tick = 0
    for t, g in d.groupby("ticker"):
        r = g["close"].pct_change().dropna()
        if len(r) < 300 or r.std() <= 0:
            continue
        s = (r - r.mean()) / r.std()
        s.index = pd.to_datetime(g["date"].iloc[1:1 + len(s)])
        port_srs = s if port_srs is None else port_srs.add(s, fill_value=0)
        n_tick += 1
    port = (port_srs / n_tick).dropna().to_numpy() if port_srs is not None else np.array([])
    port_rows = []
    for k, emp, gauss, ratio in tail_probs(port):
        port_rows.append({
            "metric": f"P(|z|>{k})", "empirical": round(emp, 6),
            "gaussian": round(gauss, 6), "ratio": round(ratio, 1),
        })
    port_alpha = hill_alpha(np.abs(port))
    pd.DataFrame(port_rows).to_csv(DATA_DIR / "portfolio_tail.csv", index=False)

    # tail dependence on top liquid names (subset to keep pairwise feasible)
    # date-aligned series so different listing histories intersect correctly
    aligned = {}
    for t, g in d.groupby("ticker"):
        r = g["close"].pct_change().dropna()
        if len(r) < 300:
            continue
        s = pd.Series(r.to_numpy(), index=pd.to_datetime(g["date"].iloc[1:1 + len(r)]))
        aligned[t] = s
    top = sorted(aligned, key=lambda t: -len(aligned[t]))[:80]
    dep_rows = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            up, lo = tail_dependence(aligned[top[i]], aligned[top[j]])
            if up is None:
                continue
            dep_rows.append({
                "pair_id": f"{top[i]}|{top[j]}", "upper_tail_dep": round(up, 3),
                "lower_tail_dep": round(lo, 3),
                "tail_dep_max": round(max(up, lo), 3),
            })
    dep = pd.DataFrame(dep_rows).sort_values("tail_dep_max", ascending=False).head(args.top_pairs)
    dep.to_csv(DATA_DIR / "tail_dependence.csv", index=False)

    print(f"tail_index.csv: {len(df)} tickers | portfolio alpha={round(port_alpha,2)}")
    print(f"portfolio_tail.csv: {len(port_rows)} rows")
    print(f"tail_dependence.csv: {len(dep)} pairs")
    if len(df):
        worst = df.sort_values("tail_ratio_5sd", ascending=False).head(5)
        print("\nWorst tail ratios (emp/gauss @5sd) — the names Gaussian risk models understate most:")
        print(worst[["ticker", "tail_ratio_5sd", "kurtosis"]].to_string(index=False))


if __name__ == "__main__":
    main()
