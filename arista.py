#!/usr/bin/env python3
"""
arista.py — ARISTA top-of-uptrend detector across the monitored universe.

ARISTA = Accelerating Rally Into Sustained Top Alert. Detects when a strong
uptrend's *slope* is rolling over at/near a high — the setup that preceded
FTNT's Jul-9-2025 peak and Aug-1-8 2025 breakdown. It is a top detector
(de-risk / trim / stand-down), not a timing tool: on FTNT it fired ~4 weeks
before the break.

Why it exists: the ride layer (shock_ride) gates ENTRY on the slope of
momentum and durability, but had no systematic companion for detecting the TOP
of an uptrend. ARISTA is that companion — the same momentum/deceleration lens
pointed at exhaustion instead of ignition. It is computed daily, no lookahead
(only data up to each date).

Components (all point-in-time, per ticker, on split-adjusted closes):

  mom3  = 63-session cumulative return
  mom6  = 126-session cumulative return
  decel = mom6 - mom3        <0  → the 6m trend slope is LESS than the 3m slope
                                (momentum decelerating / rolling over)
  downshare = 20d down-dollar-volume / (down+up dollar-volume)
                             rising toward/above 0.5 → distribution (sellers)
  from20 = close / 20d-high - 1    <0 → failing to make new highs
  at_year_high = close / 252d-high       near 1.0 → strong long-term trend context

ARISTA legs (separately scored, then combined):

  leg_divergence = -decel                (momentum divergence at high; the LEAD)
  leg_distribution = downshare - 0.5
  leg_rollover = -from20
  leg_high = at_year_high

ARISTA signal (the actionable top):
  arista_signal = (at_year_high > 0.92) & (decel < -0.05)
                  momentum diverging while still within ~8% of the 1-yr high.

ARISTA score (0..1-ish intensity) = weighted combination:
  score = wd*leg_divergence + ws*leg_distribution + wr*leg_rollover + wh*leg_high
  normalized so a clear top (FTNT 7/2025) scores ~0.8+.

Outputs:
  arista_metrics.parquet   — FULL daily metrics time series for every ticker
                             (date, ticker, close, mom3, mom6, decel,
                              downshare, from20, at_year_high, leg_*,
                              arista_score, arista_signal). This is the
                             backtesting surface — join on (ticker,date) to
                             test the detector across the whole universe.
  arista_signals.parquet   — LATEST snapshot per ticker (date + last-row
                             metrics + signal + interpretation).
  arista_backtest.parquet  — summary of signal→forward-return performance per
                             ticker and in aggregate (honest measured stats).

Usage:
  python arista.py [--save] [--tickers AAPL,MSFT]
  python arista.py backtest [--save]   # print + persist aggregate backtest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_common import DATA_DIR, load_adj_prices_pandas

OUT_METRICS = DATA_DIR / "arista_metrics.parquet"
OUT_SIGNALS = DATA_DIR / "arista_signals.parquet"
OUT_BACKTEST = DATA_DIR / "arista_backtest.parquet"

# Decel window pair (6m vs 3m) in sessions.
M6 = 126
M3 = 63
DIST_WINDOW = 20      # down-volume-share & rollover window
YEAR_WINDOW = 252     # 1-yr high proximity window
SIGNAL_DECEL = -0.05  # decel below this = momentum diverging
SIGNAL_HIGH = 0.92    # close above this fraction of 1-yr high


def _down_volume_share(close: pd.Series, volume: pd.Series | None, window: int = DIST_WINDOW) -> pd.Series:
    """20d down-dollar-volume / (down+up dollar-volume). NaN if no volume."""
    if volume is None or volume.dropna().empty:
        return pd.Series(np.nan, index=close.index)
    ret = close.pct_change()
    dv = (volume * ret.clip(upper=0).abs()).rolling(window).sum()
    uv = (volume * ret.clip(lower=0)).rolling(window).sum()
    denom = dv + uv
    return (dv / denom.replace(0.0, np.nan)).fillna(0.5)


def compute_metrics(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    """Full point-in-time ARISTA metrics series for one ticker."""
    c = close.dropna()
    if len(c) < 40:
        return pd.DataFrame()
    ret = c.pct_change()
    # Momentum = price-ratio over the window (vectorized; == cumulative-return
    # of the window). Rolling apply was too slow across the universe.
    mom3 = c / c.shift(M3) - 1
    mom6 = c / c.shift(M6) - 1
    decel = mom6 - mom3
    downshare = _down_volume_share(c, volume)
    hi20 = c.rolling(DIST_WINDOW).max()
    from20 = c / hi20 - 1
    hi_year = c.rolling(YEAR_WINDOW).max()
    at_year = c / hi_year

    df = pd.DataFrame({
        "close": c,
        "mom3": mom3,
        "mom6": mom6,
        "decel": decel,
        "downshare": downshare,
        "from20": from20,
        "at_year_high": at_year,
    })

    df["leg_divergence"] = (-decel).clip(lower=0)
    df["leg_distribution"] = (downshare - 0.5).clip(lower=0)
    df["leg_rollover"] = (-from20).clip(lower=0)

    # Normalize each leg to [0,1] with fixed, interpretable caps, then combine.
    #  - divergence: decel of -15pp (6m return 15pp below 3m return) = full.
    #  - distribution: downshare 0.80 (heavy selling) = full.
    #  - rollover: 12% off the 20d high = full.
    # A genuine top (FTNT 7/2025: decel -0.13, rollover starting, distribution
    # rising) scores ~0.7-0.9. A healthy accelerating uptrend scores near 0.
    div_n = (df["leg_divergence"] / 0.15).clip(0, 1)
    dist_n = (df["leg_distribution"] / 0.30).clip(0, 1)
    roll_n = (df["leg_rollover"] / 0.12).clip(0, 1)
    # High-proximity is a CONTEXT gate (a top only matters near a high), used as
    # a multiplier ~0.8-1.0 rather than an additive component that saturates.
    high_n = df["at_year_high"].clip(0.80, 1.0)
    df["arista_score"] = (high_n * (0.45 * div_n + 0.30 * dist_n + 0.25 * roll_n)).clip(0, 1)

    df["arista_signal"] = (df["at_year_high"] > SIGNAL_HIGH) & (df["decel"] < SIGNAL_DECEL)
    return df


def _interpret(r: pd.Series) -> str:
    if bool(r["arista_signal"]):
        return (f"TOP-LIKE: momentum diverging (6m-3m {r['decel']:+.0%}) "
                f"while near 1-yr high ({r['at_year_high']:.2f}). De-risk / trim / stand down.")
    if r["arista_score"] > 0.5:
        return (f"ELEVATED (score {r['arista_score']:.2f}): "
                f"decel {r['decel']:+.0%}, downshare {r['downshare']:.2f}, "
                f"from 20d high {r['from20']:+.0%}. Watch for confirmation.")
    return (f"BUILDING/CLEAR (score {r['arista_score']:.2f}): "
            f"decel {r['decel']:+.0%}, downshare {r['downshare']:.2f}, "
            f"at 1-yr high {r['at_year_high']:.2f}.")


def build(tickers: list[str] | None = None) -> pd.DataFrame:
    """Return long-form metrics (date,ticker,...) for the universe."""
    prices = load_adj_prices_pandas()  # date,ticker,close (adj_close renamed)
    if tickers:
        prices = prices[prices["ticker"].isin(tickers)]
    # volume for distribution leg
    vol = {}
    try:
        vp = pd.read_parquet(DATA_DIR / "daily_prices.parquet", columns=["date", "ticker", "volume"])
        vp["date"] = pd.to_datetime(vp["date"])
        for tk, g in vp.groupby("ticker"):
            vol[tk] = g.set_index("date")["volume"]
    except Exception:
        vol = {}

    frames = []
    for tk, g in prices.groupby("ticker"):
        g = g.sort_values("date").set_index("date")
        v = vol.get(tk)
        m = compute_metrics(g["close"], v)
        if m.empty:
            continue
        m.insert(0, "ticker", tk)
        frames.append(m.reset_index().rename(columns={"date": "date"}))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def latest_signals(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker = most recent date's metrics + signal."""
    if metrics.empty:
        return pd.DataFrame()
    idx = metrics.groupby("ticker")["date"].idxmax()
    out = metrics.loc[idx].copy()
    out["interpretation"] = out.apply(_interpret, axis=1)
    return out


def _fwd_dd_array(close: np.ndarray, window: int) -> np.ndarray:
    """For each i, the max drawdown over the forward window close[i:i+window].

    Reference peak = the highest close reached in [i, i+window); drawdown =
    lowest close in the window vs that peak. This correctly captures a top that
    runs up a little then falls (the reference peak is the post-signal high),
    and is O(n) via reversed rolling max/min. NaN where the window is short.

    close: numpy array, last = newest.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < 5 or window < 1:
        return out
    import pandas as pd
    rev = close[::-1].copy()
    # rolling on reversed array: fwd_max[i] = max over original [i, i+window)
    fwd_max = pd.Series(rev).rolling(window, min_periods=1).max().to_numpy()[::-1]
    fwd_min = pd.Series(rev).rolling(window, min_periods=1).min().to_numpy()[::-1]
    valid = ~np.isnan(fwd_max) & ~np.isnan(fwd_min) & (fwd_max > 0)
    dd = np.where(valid, fwd_min / np.where(fwd_max == 0, np.nan, fwd_max) - 1, np.nan)
    # Require a meaningful forward window (>=5 sessions) before the value counts.
    out[:n - 5 + 1] = dd[:n - 5 + 1]
    return out


def backtest(metrics: pd.DataFrame, close_by: dict = None, max_days: int = 120) -> pd.DataFrame:
    """Honest forward-return stats after arista_signal.

    For each signal session, measure the forward max drawdown and forward
    return over `max_days`, and whether a >=15% drawdown followed within that
    window (a "caught top"). Per-ticker aggregate + grand total.

    Vectorized per ticker: a single forward-drawdown array is built once and
    indexed at signal positions (fast across the full universe).
    """
    if metrics.empty:
        return pd.DataFrame(metrics)
    rows = []
    for tk, g in metrics.groupby("ticker"):
        cs = close_by.get(tk) if close_by else None
        if cs is None or not cs.index.is_monotonic_increasing:
            continue
        sigs = g[g["arista_signal"]].sort_values("date")
        if sigs.empty:
            continue
        arr = cs.to_numpy()
        idx = cs.index.to_numpy()
        pos = np.searchsorted(idx, sigs["date"].to_numpy(dtype="datetime64[ns]"), side="right")
        fwd_dd = _fwd_dd_array(arr, max_days)
        for i, (_, srow) in zip(pos, sigs.iterrows()):
            p = int(i)
            if p >= len(fwd_dd):
                continue
            mdd = fwd_dd[p]
            if np.isnan(mdd):
                continue
            hit = mdd <= -0.15
            fwd_ret = np.nan
            hi = p + max_days
            if hi < len(arr) and arr[p] and not np.isnan(arr[p]):
                fwd_ret = float(arr[hi] / arr[p] - 1)
            rows.append({
                "ticker": tk,
                "signal_date": srow["date"],
                "signal_score": float(srow["arista_score"]),
                "fwd_max_drawdown": float(mdd),
                "fwd_return": fwd_ret,
                "caught_15pct_top": bool(hit),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    per = df.groupby("ticker").agg(
        n_signals=("signal_score", "size"),
        mean_score=("signal_score", "mean"),
        avg_fwd_dd=("fwd_max_drawdown", "mean"),
        avg_fwd_ret=("fwd_return", "mean"),
        caught_rate=("caught_15pct_top", "mean"),
    ).reset_index()
    total = pd.DataFrame([{
        "ticker": "TOTAL",
        "n_signals": int(len(df)),
        "mean_score": float(df["signal_score"].mean()),
        "avg_fwd_dd": float(df["fwd_max_drawdown"].mean()),
        "avg_fwd_ret": float(df["fwd_return"].mean()) if df["fwd_return"].notna().any() else np.nan,
        "caught_rate": float(df["caught_15pct_top"].mean()),
    }])
    per = pd.concat([per, total], ignore_index=True)
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=None, help="comma list; default = all with prices")
    ap.add_argument("--save", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sb = sub.add_parser("backtest", help="print + persist aggregate backtest stats")
    sb.add_argument("--save", action="store_true")
    sb.set_defaults(cmd="backtest")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    metrics = build(tickers)
    if metrics.empty:
        print("No metrics computed.")
        return 1

    # close_by for backtest
    close_by = {tk: g.set_index("date")["close"] for tk, g in
                load_adj_prices_pandas().groupby("ticker")}

    signals = latest_signals(metrics)

    if args.cmd == "backtest":
        bt = backtest(metrics, close_by)
        print(f"=== ARISTA backtest (signal -> forward {120}d) — {len(metrics.groupby('ticker'))} tickers ===")
        if not bt.empty:
            tot = bt[bt["ticker"] == "TOTAL"]
            if not tot.empty:
                t = tot.iloc[0]
                print(f"  TOTAL: {int(t['n_signals'])} signals | avg fwd maxDD {t['avg_fwd_dd']:+.0%} "
                      f"| avg fwd return {t['avg_fwd_ret']:+.0%} | caught >=15% top {t['caught_rate']:.0%}")
            per = bt[bt["ticker"] != "TOTAL"].nlargest(15, "n_signals")
            print(per.to_string(index=False))
        if args.save:
            bt.to_parquet(OUT_BACKTEST)
            print(f"\nWrote {OUT_BACKTEST}")

    # print latest signals (top scoring first)
    print(f"\n=== ARISTA latest signals ({len(signals)} tickers) — top by score ===")
    cols = [c for c in ["ticker", "date", "close", "decel", "downshare", "from20",
                        "at_year_high", "arista_score", "arista_signal"] if c in signals]
    top = signals.nlargest(15, "arista_score")[cols]
    print(top.to_string(index=False))

    if args.save:
        metrics.to_parquet(OUT_METRICS)
        signals.to_parquet(OUT_SIGNALS)
        print(f"\nWrote {OUT_METRICS} ({len(metrics)} rows)")
        print(f"Wrote {OUT_SIGNALS} ({len(signals)} rows)")
    return 0


if __name__ == "__main__":
    exit(main())
