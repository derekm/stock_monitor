#!/usr/bin/env python3
"""statistical_profiler.py — a TRUE statistical profiler over fractal windows,
persisting a wide per-window statistic set so diverse experiments run on saved
results instead of re-running price history.

The fractal-of-sliding-windows scheme (US20120253946A1) tiles a range into every
aligned granularity; the existing `fractal_signal_vec` computes only momentum
(log-ret, slope, ret/vol). This module widens that into a full statistical
profile of EACH trailing fractal window:

  PRICE statistics (on close):
    price_mean, price_median, price_mode, price_max, price_min, price_range
    price_skew, price_kurtosis, price_std
    close_z        — z-score of the current close vs the window distribution
    close_pctile   — percentile rank of the current close within the window
    runup          — current close vs window min (fraction of range travelled)
    window_drawdown— max drawdown of the window from its own running peak
    price_slope    — linear slope of price over the window (dollars/day)
    price_curvature— 2nd-order coefficient (concavity) of price over the window

  VOLUME statistics (on volume, which is universally present):
    volume_mean, vwap (sum(close*vol)/sum(vol) — close-based approximation since
    high/low are ~absent), volume_z (volume vs its own window distribution)

  MOMENTUM statistics (existing, kept for comparison):
    log_ret (window log return), momentum (log_ret / vol), ret_vol (window vol)

The `date` in each row is the window END date; each span covers the TRAILING
`span_len` trading days ending at that date. All stats are point-in-time (only
data up to the window end).

Persistence:
  statistical_profiler.py --save   writes long-format `fractal_profiles.parquet`
  keyed by (ticker, date, span_len, span_from, span_to) with all stats above,
  computed once. Downstream experiments (backtest_price_vs_momentum.py) read this
  parquet instead of recomputing window statistics.

Usage: python statistical_profiler.py [--tickers N] [--window 1500] [--save]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
OUT = DATA_DIR / "fractal_profiles.parquet"
# granularity ladder. (3,5) = 15-day full window at base-3 granularity (15 spans
# of lengths 3,6,9,12,15) — the finest, quickest view. Then (5,3), (10,3),
# (15,3), (30,3) scale up to 15d/30d/45d/90d coarser full windows.
CONFIGS = [(3, 5), (5, 3), (10, 3), (15, 3), (30, 3)]
MIN_DAYS = 60

# stats computed per window (kept in a stable order for schema documentation)
# The trailing *V columns (vwap_true, atr, atr_pct, gap_mean, gap_std, range_hl,
# body_mean, body_std, upper_wick, lower_wick) are true-OHLCV stats: they need
# open/high/low and are NaN where those are missing (old close-only history).
STAT_COLS = [
    "price_mean", "price_median", "price_mode", "price_max", "price_min",
    "price_range", "price_std", "price_skew", "price_kurtosis",
    "close_z", "close_pctile", "runup", "window_drawdown",
    "price_slope", "price_curvature",
    "volume_mean", "vwap", "volume_z",
    "log_ret", "momentum", "ret_vol",
    # true-OHLCV stats (require open/high/low; NaN on close-only history)
    "vwap_true", "atr", "atr_pct", "gap_mean", "gap_std", "range_hl",
    "body_mean", "body_std", "upper_wick", "lower_wick",
]


def spans_configs():
    """All (span_from, span_to, span_len) for the granularity ladder."""
    from fractal_windows import spans_generator
    out = []
    for a, b in CONFIGS:
        for f, t in spans_generator(a, b):
            out.append((f, t, t - f))
    # dedupe by length is NOT wanted here — keep all (f,t) so experiments can
    # separate the base granularity from the full-window. Return unique (f,t,len).
    seen = set()
    uniq = []
    for f, t, L in out:
        if (f, t) not in seen:
            seen.add((f, t))
            uniq.append((f, t, L))
    return uniq


def _rolling_mode(close: np.ndarray, L: int, n_bins: int = 9) -> np.ndarray:
    """Approximate rolling mode of the trailing L-window via a fixed histogram.

    The window is normalized by its running max (so bins are price-ratio
    invariant), values are assigned to `n_bins` log-spaced bins, and the mode is
    the bin center that is most frequently occupied. Vectorized over time.
    Returns array of mode values (NaN before L-1).
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < L:
        return out
    # normalized log-price in window; bin edges span the window range
    logc = np.log(np.where(close > 0, close, np.nan))
    # use rolling min/max to normalize
    roll_min = pd.Series(close).rolling(L).min().to_numpy()
    roll_max = pd.Series(close).rolling(L).max().to_numpy()
    span = roll_max - roll_min
    ok = np.isfinite(span) & (span > 0)
    # relative position 0..1 of each close in its window
    rel = np.where(ok, (close - roll_min) / np.where(span > 0, span, 1.0), np.nan)
    bins = np.arange(n_bins + 1) / n_bins
    idx = np.digitize(rel, bins) - 1  # 0..n_bins-1
    idx = np.where(np.isfinite(rel), idx, -1)
    # rolling histogram: count of each bin over the trailing L window
    nrow = n
    onehot = np.zeros((nrow, n_bins))
    valid = idx >= 0
    for b in range(n_bins):
        onehot[valid & (idx == b), b] = 1.0
    roll_hist = pd.DataFrame(onehot).rolling(L).sum().to_numpy()  # [n, n_bins]
    best = np.argmax(roll_hist, axis=1)
    # mode = lower edge + (best+0.5)/n_bins of the window range
    frac = (best + 0.5) / n_bins
    out = np.where(ok, roll_min + span * frac, np.nan)
    # zero out rows before the window is fully formed
    out[: L - 1] = np.nan
    return out


def _rolling_skew_kurt(x: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling skewness and kurtosis via moment rolling sums (vectorized)."""
    s = pd.Series(x)
    n = s.rolling(L).count().to_numpy()
    m1 = s.rolling(L).mean().to_numpy()
    d2 = (s - s.rolling(L).mean()) ** 2
    m2 = d2.rolling(L).mean().to_numpy()
    d3 = ((s - s.rolling(L).mean()) ** 3).rolling(L).mean().to_numpy()
    d4 = ((s - s.rolling(L).mean()) ** 4).rolling(L).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.where(n >= 3, d3 / np.power(m2, 1.5), np.nan)
        kurt = np.where(n >= 4, d4 / np.square(m2) - 3.0, np.nan)
    return skew, kurt


def window_profile_stats(close: pd.Series, volume: pd.Series | None,
                         L: int, open_: pd.Series | None = None,
                         high: pd.Series | None = None,
                         low: pd.Series | None = None) -> pd.DataFrame:
    """Full statistical profile of every trailing L-day window.

    close/volume: DatetimeIndexed daily series. If open_/high/low are given,
    true-OHLCV statistics are computed (real VWAP via typical price, ATR,
    gap, candle-body and wick shape); otherwise those columns are NaN.
    Returns a DataFrame indexed by the window END date with one row per day
    and one column per STAT_COLS entry. All point-in-time.
    """
    c = close.to_numpy(dtype=float)
    n = len(c)
    logc = np.log(np.where(c > 0, c, np.nan))
    idx = np.arange(n, dtype=float)
    # price stats via rolling
    pm = pd.Series(c).rolling(L).mean().to_numpy()
    pmed = pd.Series(c).rolling(L).median().to_numpy()
    pstd = pd.Series(c).rolling(L).std().to_numpy()
    pmax = pd.Series(c).rolling(L).max().to_numpy()
    pmin = pd.Series(c).rolling(L).min().to_numpy()
    prange = pmax - pmin
    pmode = _rolling_mode(c, L)
    pskew, pkurt = _rolling_skew_kurt(c, L)
    # close z-score & percentile rank within window
    with np.errstate(divide="ignore", invalid="ignore"):
        close_z = np.where(pstd > 0, (c - pm) / pstd, np.nan)
    # percentile rank: fraction of window values <= current close
    # (rolling rank approximation via counting below current)
    pctile = np.full(n, np.nan)
    for i in range(L - 1, n):
        w = c[i - L + 1:i + 1]
        pctile[i] = float((w <= c[i]).mean())
    # runup: current close as fraction of [min,max] range travelled
    runup = np.where(prange > 0, (c - pmin) / np.where(prange > 0, prange, 1.0), np.nan)
    # window drawdown: current close vs running peak of the window
    cummax_w = pd.Series(c).rolling(L).max().to_numpy()  # rolling max is the window peak
    window_dd = c / np.where(cummax_w > 0, cummax_w, np.nan) - 1.0
    # price slope & curvature via closed-form OLS on price vs time over window
    sy = pd.Series(c).rolling(L).sum().to_numpy()
    sky = pd.Series(idx * c).rolling(L).sum().to_numpy()
    start = idx - (L - 1)
    sxy = sky - start * sy
    sx = L * (L - 1) / 2.0
    sxx = L * (L - 1) * (2 * L - 1) / 6.0
    denom = L * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        price_slope = (L * sxy - sx * sy) / denom
        # curvature: OLS on price vs t^2 (2nd-order) — use regression of y on t and t^2
    # curvature via 2x2 normal equations
    sx2 = np.arange(n) ** 2
    sx3 = np.arange(n) ** 3
    sx4 = np.arange(n) ** 4
    rsx = pd.Series(sx2).rolling(L).sum().to_numpy()
    rsx3 = pd.Series(sx3).rolling(L).sum().to_numpy()
    rsx4 = pd.Series(sx4).rolling(L).sum().to_numpy()
    rsy = sy
    rsxy = pd.Series(idx * c).rolling(L).sum().to_numpy()
    rsx2y = pd.Series(sx2 * c).rolling(L).sum().to_numpy()
    n2 = np.full(n, float(L))
    A = np.zeros((n, 3, 3)); B = np.zeros((n, 3))
    A[:, 0, 0] = n2; A[:, 0, 1] = sx; A[:, 0, 2] = rsx
    A[:, 1, 0] = sx; A[:, 1, 1] = rsx; A[:, 1, 2] = rsx3
    A[:, 2, 0] = rsx; A[:, 2, 1] = rsx3; A[:, 2, 2] = rsx4
    B[:, 0] = rsy; B[:, 1] = rsxy; B[:, 2] = rsx2y
    coef = np.zeros((n, 3))
    for i in range(L - 1, n):
        try:
            coef[i] = np.linalg.solve(A[i], B[i])
        except np.linalg.LinAlgError:
            coef[i] = [np.nan] * 3
    curvature = coef[:, 2]
    price_slope_fit = coef[:, 1]

    # volume stats
    if volume is not None:
        v = volume.to_numpy(dtype=float)
        vmean = pd.Series(v).rolling(L).mean().to_numpy()
        vstd = pd.Series(v).rolling(L).std().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            volume_z = np.where(vstd > 0, (v - vmean) / vstd, np.nan)
        # vwap = sum(close*vol)/sum(vol)  (close-based approximation)
        vwap = (pd.Series(c * v).rolling(L).sum() / pd.Series(v).rolling(L).sum()).to_numpy()
    else:
        vmean = np.full(n, np.nan); volume_z = np.full(n, np.nan); vwap = np.full(n, np.nan)

    # ── true-OHLCV stats (only where open/high/low present) ───────────────
    has_ohlc = (open_ is not None and high is not None and low is not None)
    if has_ohlc:
        o = open_.reindex(close.index).to_numpy(dtype=float)
        h = high.reindex(close.index).to_numpy(dtype=float)
        lo = low.reindex(close.index).to_numpy(dtype=float)
        v = volume.to_numpy(dtype=float) if volume is not None else np.ones(n)
        # typical price = (H+L+C)/3 ; true VWAP = sum(typ*vol)/sum(vol)
        typ = (h + lo + c) / 3.0
        vwap_true = (pd.Series(typ * v).rolling(L).sum() / pd.Series(v).rolling(L).sum()).to_numpy()
        # true range = max(H-L, |H-prevC|, |L-prevC|)
        prev_c = np.roll(c, 1); prev_c[0] = np.nan
        tr = np.maximum.reduce([h - lo, np.abs(h - prev_c), np.abs(lo - prev_c)])
        atr = pd.Series(tr).rolling(L).mean().to_numpy()
        atr_pct = atr / np.where(c > 0, c, np.nan)  # ATR as fraction of price
        # gap = open vs previous close (fractional)
        gap = o / np.where(prev_c > 0, prev_c, np.nan) - 1.0
        gap_mean = pd.Series(gap).rolling(L).mean().to_numpy()
        gap_std = pd.Series(gap).rolling(L).std().to_numpy()
        # intraday range (H-L)/C
        range_hl = pd.Series((h - lo) / np.where(c > 0, c, np.nan)).rolling(L).mean().to_numpy()
        # candle body (C-O)/C and wicks
        body = (c - o) / np.where(c > 0, c, np.nan)
        body_mean = pd.Series(body).rolling(L).mean().to_numpy()
        body_std = pd.Series(body).rolling(L).std().to_numpy()
        up_wick = (h - np.maximum(c, o)) / np.where(c > 0, c, np.nan)
        lo_wick = (np.minimum(c, o) - lo) / np.where(c > 0, c, np.nan)
        upper_wick = pd.Series(up_wick).rolling(L).mean().to_numpy()
        lower_wick = pd.Series(lo_wick).rolling(L).mean().to_numpy()
    else:
        vwap_true = atr = atr_pct = gap_mean = gap_std = np.full(n, np.nan)
        range_hl = body_mean = body_std = upper_wick = lower_wick = np.full(n, np.nan)

    # momentum stats (existing)
    log_ret = logc - np.concatenate([np.full(L, np.nan), logc[:-L]])
    ret_vol = pd.Series(np.diff(logc, prepend=np.nan)).rolling(L).std().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum = log_ret / np.where(ret_vol > 0, ret_vol, np.nan)

    df = pd.DataFrame(index=close.index)
    df["close"] = c
    df["price_mean"] = pm
    df["price_median"] = pmed
    df["price_mode"] = pmode
    df["price_max"] = pmax
    df["price_min"] = pmin
    df["price_range"] = prange
    df["price_std"] = pstd
    df["price_skew"] = pskew
    df["price_kurtosis"] = pkurt
    df["close_z"] = close_z
    df["close_pctile"] = pctile
    df["runup"] = runup
    df["window_drawdown"] = window_dd
    df["price_slope"] = price_slope_fit
    df["price_curvature"] = curvature
    df["volume_mean"] = vmean
    df["vwap"] = vwap
    df["volume_z"] = volume_z
    df["log_ret"] = log_ret
    df["momentum"] = momentum
    df["ret_vol"] = ret_vol
    # true-OHLCV stats
    df["vwap_true"] = vwap_true
    df["atr"] = atr
    df["atr_pct"] = atr_pct
    df["gap_mean"] = gap_mean
    df["gap_std"] = gap_std
    df["range_hl"] = range_hl
    df["body_mean"] = body_mean
    df["body_std"] = body_std
    df["upper_wick"] = upper_wick
    df["lower_wick"] = lower_wick
    return df[["close"] + STAT_COLS]


def profile_ticker(close: pd.Series, volume: pd.Series | None,
                   spans: list[tuple[int, int, int]],
                   open_: pd.Series | None = None,
                   high: pd.Series | None = None,
                   low: pd.Series | None = None) -> pd.DataFrame:
    """Full long-format statistical profile of one ticker across all spans.

    Returns DataFrame: date, span_from, span_to, span_len, + all STAT_COLS.
    """
    frames = []
    lens = sorted({L for _, _, L in spans})
    for L in lens:
        stats = window_profile_stats(close, volume, L, open_=open_, high=high, low=low)
        for f, t, slen in spans:
            if slen != L:
                continue
            fr = stats.copy()
            fr["span_from"] = f
            fr["span_to"] = t
            fr["span_len"] = L
            frames.append(fr.reset_index().rename(columns={"index": "date"}))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[["date", "span_from", "span_to", "span_len", "close"] + STAT_COLS]
    return out


def build_profiles(tickers_cap: int | None = None, window: int = 1500,
                   tickers_list: list[str] | None = None) -> pd.DataFrame:
    """Compute profiles for a universe (or explicit ticker list)."""
    from macro_sector_shock import _load_price_matrix, _price_universe
    w = _load_price_matrix()
    have = _price_universe()
    tickers = sorted(have & set(w.columns))
    if tickers_list:
        tickers = [t for t in tickers_list if t in set(w.columns)]
    if tickers_cap:
        tickers = tickers[:tickers_cap]

    # OHLCV matrices
    vp = pd.read_parquet(DATA_DIR / "daily_prices.parquet",
                         columns=["date", "ticker", "volume", "open", "high", "low"])
    vp["date"] = pd.to_datetime(vp["date"])
    # Reindex OHLCV to complete business-day calendar, then forward-fill
    # so holiday dates get the prior trading day's values
    all_dates = pd.date_range(vp["date"].min(), vp["date"].max(), freq="B")
    vp = vp.pivot(index="date", columns="ticker", values=["volume", "open", "high", "low"])
    vp = vp.reindex(all_dates)
    vp = vp.ffill()
    vp = vp.stack().reset_index()
    vp.columns = ["date", "ticker", "volume", "open", "high", "low"]
    vm = vp.pivot(index="date", columns="ticker", values="volume")
    om = vp.pivot(index="date", columns="ticker", values="open")
    hm = vp.pivot(index="date", columns="ticker", values="high")
    lm = vp.pivot(index="date", columns="ticker", values="low")

    spans = spans_configs()
    frames = []
    for t in tickers:
        c = w[t].dropna()
        if len(c) < MIN_DAYS:
            continue
        c = c.tail(window)
        vol = vm[t].reindex(c.index).ffill() if t in vm.columns else None
        op = om[t].reindex(c.index) if t in om.columns else None
        hi = hm[t].reindex(c.index) if t in hm.columns else None
        lo = lm[t].reindex(c.index) if t in lm.columns else None
        pf = profile_ticker(c, vol, spans, open_=op, high=hi, low=lo)
        if pf.empty:
            continue
        pf["ticker"] = t
        frames.append(pf)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=None, help="cap universe size")
    ap.add_argument("--window", type=int, default=1500, help="trailing days per ticker")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    print(f"Building statistical profiles (window={args.window})...")
    df = build_profiles(args.tickers, args.window)
    print(f"  rows: {len(df)} | tickers: {df['ticker'].nunique()} | spans: {df['span_len'].nunique()}")
    print(f"  stat cols: {len(STAT_COLS)}")

    if args.save and len(df):
        df.to_parquet(OUT, index=False)
        print(f"Wrote {OUT} ({len(df)} rows, {df['ticker'].nunique()} tickers)")
    # quick sanity: sample RAL
    ral = df[df["ticker"] == "RAL"]
    if len(ral):
        last = ral[ral["span_len"] == 90].sort_values("date").iloc[-1]
        print("\nSample RAL 90d window profile (most recent):")
        print(last[["date", "span_from", "span_to", "price_mean", "price_median",
                    "price_mode", "vwap", "close_z", "close_pctile", "runup",
                    "window_drawdown", "log_ret", "momentum"]].to_string())
    return 0


if __name__ == "__main__":
    exit(main())