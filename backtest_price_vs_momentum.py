#!/usr/bin/env python3
"""backtest_price_vs_momentum.py — do PRICE-based fractal signals predict
forward returns better than MOMENTUM-based ones?

Reads the persisted `fractal_profiles.parquet` (written by
`statistical_profiler.py`) and compares the predictive power of a wide array of
per-window statistics against forward monthly returns. This runs on SAVED
profiles — it never recomputes window statistics from raw prices.

Signal families (each a 0/1 feature, tested against 3/6/12-month forward log
returns):

  MOMENTUM (existing)   : log_ret>0, momentum>0, momentum>median, frac-based
  PRICE POSITION        : close_z>0, close_pctile>0.5, close_pctile>0.8,
                          runup>0.5, close>price_median, close>price_mean
  PRICE STRUCTURE       : price_skew>0, price_curvature>0 (concave up), window
                          drawdown shallow (window_drawdown>-0.1)
  VOLUME                : volume_z>0, close>vwap
  HYBRID                : close>vwap AND momentum>0; pctile>0.8 AND momentum>0

For each feature at each horizon we compute:
  hit_rate_on  — fraction of signal-on periods with positive forward return
  mean_on / mean_off / spread — mean forward return when on vs off
  annual_spread— spread annualized to the horizon
  base_mean    — unconditional mean forward return

The headline question: does the best PRICE signal beat the best MOMENTUM signal
on annualized spread? (From the original fractal backtest, momentum-consensus
hit ~0.64-0.65 but did NOT beat the best single window; here we test whether a
price-position signal like `close>median` or `runup>0.5` is more informative.)

Usage: python backtest_price_vs_momentum.py [--horizon 3,6,12] [--min-n 100]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
PROFILES = DATA_DIR / "fractal_profiles.parquet"
OUT = DATA_DIR / "backtest_price_vs_momentum.parquet"


def build_forward(df: pd.DataFrame, horizon_days: int, price_matrix=None) -> pd.DataFrame:
    """Attach forward DAILY log return over `horizon_days` trading days to each row.

    df: long-format profiles (ticker, date, span, ...). For each (ticker, date)
    we find the close `horizon_days` trading days later on the DAILY price series
    and compute the log return. Returns df with an added `fwd_log_ret` column
    (NaN where the forward price is not yet available). Vectorized via per-ticker
    positional lookups on the daily index — NO monthly resampling.
    """
    if price_matrix is None:
        price_matrix = _load_prices()
    # per-ticker daily close + log-price arrays
    luts = {}
    for t in df["ticker"].unique():
        if t not in price_matrix.columns:
            continue
        close = price_matrix[t].dropna()
        if close.notna().sum() < horizon_days + 2:
            continue
        lp = np.log(close.values)
        luts[t] = {"dates": close.index.values, "lp": lp}
    if not luts:
        return pd.DataFrame()

    tickers = df["ticker"].values
    dates = pd.to_datetime(df["date"]).values.astype("datetime64[ns]")
    fwd = np.full(len(df), np.nan)
    # group by ticker ONCE (pandas groupby is O(N) total), then vectorize per group
    groups = pd.Series(np.arange(len(df)), index=df.index).groupby(df["ticker"]).indices
    for t, idx in groups.items():
        lu = luts.get(t)
        if lu is None:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        pos = np.searchsorted(lu["dates"], dates[idx], side="right") - 1
        pos = np.clip(pos, 0, len(lu["lp"]) - 1)
        end = pos + horizon_days
        ok = end < len(lu["lp"])
        fwd[idx[ok]] = lu["lp"][end[ok]] - lu["lp"][pos[ok]]
    out = df.copy()
    out["fwd_log_ret"] = fwd
    return out


def _load_prices() -> pd.DataFrame:
    from macro_sector_shock import _load_price_matrix
    return _load_price_matrix()


FEATURES = {
    # momentum (existing)
    "mom:log_ret>0": lambda r: r["log_ret"] > 0,
    "mom:momentum>0": lambda r: r["momentum"] > 0,
    "mom:momentum>median": lambda r: r["momentum"] > r["momentum"].median(),
    # price position
    "price:close_z>0": lambda r: r["close_z"] > 0,
    "price:close>median": lambda r: r["close"] > r["price_median"],
    "price:close>mean": lambda r: r["close"] > r["price_mean"],
    "price:close_z>0.5": lambda r: r["close_z"] > 0.5,
    "price:pctile>0.5": lambda r: r["close_pctile"] > 0.5,
    "price:pctile>0.8": lambda r: r["close_pctile"] > 0.8,
    "price:runup>0.5": lambda r: r["runup"] > 0.5,
    "price:runup>0.8": lambda r: r["runup"] > 0.8,
    # price structure
    "struct:skew>0": lambda r: r["price_skew"] > 0,
    "struct:curv>0": lambda r: r["price_curvature"] > 0,
    "struct:dd>-0.1": lambda r: r["window_drawdown"] > -0.1,
    # volume
    "vol:volume_z>0": lambda r: r["volume_z"] > 0,
    "vol:close>vwap": lambda r: r["close"] > r["vwap"],
    "vol:vwap>median": lambda r: r["vwap"] > r["vwap"].median(),
    # hybrid price+momentum
    "hybrid:pctile>0.5&mom>0": lambda r: (r["close_pctile"] > 0.5) & (r["momentum"] > 0),
    "hybrid:pctile>0.8&mom>0": lambda r: (r["close_pctile"] > 0.8) & (r["momentum"] > 0),
    "hybrid:runup>0.5&mom>0": lambda r: (r["runup"] > 0.5) & (r["momentum"] > 0),
}

FAMILY = {
    "mom": "momentum", "price": "price", "struct": "price_structure",
    "vol": "volume", "hybrid": "hybrid",
}


def report(df: pd.DataFrame, min_n: int = 100) -> pd.DataFrame:
    out = []
    for h in sorted(df["horizon"].unique()):
        sub = df[df["horizon"] == h].dropna(subset=["fwd_log_ret"])
        if sub.empty:
            continue
        base = sub["fwd_log_ret"].mean()
        for name, fn in FEATURES.items():
            try:
                mask = fn(sub)
            except Exception:
                continue
            on = sub[mask]
            off = sub[~mask]
            if len(on) < min_n or len(off) < min_n:
                continue
            # annualize: horizon h trading days -> h/252 years
            ann = 252.0 / h
            out.append({
                "feature": name,
                "family": FAMILY.get(name.split(":")[0], "other"),
                "horizon": h,
                "n_on": len(on),
                "hit_rate_on": round((on["fwd_log_ret"] > 0).mean(), 3),
                "mean_on": round(on["fwd_log_ret"].mean(), 5),
                "mean_off": round(off["fwd_log_ret"].mean(), 5),
                "spread": round(on["fwd_log_ret"].mean() - off["fwd_log_ret"].mean(), 5),
                "annual_spread": round((on["fwd_log_ret"].mean() - off["fwd_log_ret"].mean()) * ann, 3),
                "base_mean": round(base, 5),
            })
    r = pd.DataFrame(out)
    return r.sort_values("annual_spread", ascending=False) if not r.empty else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="21,63,126", help="forward trading-day horizons")
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--span", type=int, default=None, help="only spans of this length (default: all)")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizon.split(",") if x]

    if not PROFILES.exists():
        raise SystemExit(f"{PROFILES} not found. Run `python statistical_profiler.py --save` first.")
    df = pd.read_parquet(PROFILES)
    df["date"] = pd.to_datetime(df["date"])
    if args.span:
        df = df[df["span_len"] == args.span]
        print(f"Filtering to span_len={args.span}")
    print(f"Profiles: {len(df)} rows, {df['ticker'].nunique()} tickers, {df['span_len'].nunique()} spans")

    frames = []
    w = _load_prices()
    need = ["ticker", "date", "close", "fwd_log_ret", "log_ret", "momentum",
            "close_z", "price_median", "price_mean", "close_pctile", "runup",
            "price_skew", "price_curvature", "window_drawdown", "volume_z", "vwap"]
    for h in horizons:
        print(f"  building forward returns (horizon={h} trading days)...")
        fd = build_forward(df, h, price_matrix=w)
        if fd.empty:
            continue
        fd["horizon"] = h
        fd = fd.dropna(subset=["fwd_log_ret"])
        frames.append(fd[need + ["horizon"]])
    if not frames:
        raise SystemExit("no forward returns computed")
    full = pd.concat(frames, ignore_index=True)
    print(f"  forward-return rows: {len(full)}")

    r = report(full, args.min_n)
    pd.set_option("display.width", 220)
    print("\n=== PRICE vs MOMENTUM fractal signal predictive power (daily forward) ===")
    print(r.to_string(index=False))
    if not r.empty:
        r.to_parquet(OUT, index=False)
        print(f"\nWrote {OUT} ({len(r)} rows)")

        # headline comparison
        print("\n=== Headline: best signal per family ===")
        best = r.sort_values("annual_spread", ascending=False).groupby("family").head(1)
        print(best[["feature", "family", "horizon", "annual_spread", "hit_rate_on"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    exit(main())