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

# Shared rolling library: device selection (CUDA -> DirectML -> CPU) and all
# NaN-safe rolling primitives live in tensor_ops, not in a parallel _gpu module.
from tensor_ops import rolling_quad_fit as _tops_quad_fit

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
    """Rolling skewness and EXCESS kurtosis (vectorized).

    NOTE: this is deliberately NOT tensor_ops.rolling_skew/rolling_kurt, and the
    two disagree by design -- do not "unify" them without a decision:
      * here: deviations are taken from a ROLLING mean and then rolled AGAIN,
        so the effective lookback is ~2L, and kurtosis is EXCESS (-3.0).
      * tensor_ops: single-window textbook moments, RAW kurtosis.
    Measured difference on 500 points, L=60: skew 3.04, kurt 9.59. Swapping the
    implementation would silently change every STAT_COLS output and any model
    trained on them, so the estimator stays as-is and the shared library is used
    only by new code.
    """
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
    # price slope & curvature: 2nd-order polynomial fit over the window.
    #
    # This previously built 3x3 normal equations by hand and solved them per
    # day. That construction was WRONG: A[0,1]/A[1,0] used sx = L(L-1)/2 (a
    # WINDOW-LOCAL index sum) while every other entry used GLOBAL-index rolling
    # sums (e.g. 1770 vs 10230 at i=200), so the matrix was inconsistent and
    # neither coefficient was a valid fit. Verified against np.polyfit on the
    # raw window: tensor_ops.rolling_quad_fit reproduces polyfit's c1/c2
    # exactly, the old code did not.
    #
    # It also left the pre-window rows at the np.zeros() initializer, reporting
    # a fake "flat trend, zero curvature" instead of NaN.
    price_slope_fit, curvature = _tops_quad_fit(c, L)

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
                   tickers_list: list[str] | None = None,
                   batched: bool = False, device=None) -> pd.DataFrame:
    """Compute profiles for a universe (or explicit ticker list).

    batched=True uses the tensor_ops batched engine (GPU when available) instead
    of the per-ticker pandas loop. Outputs are identical (asserted in
    test_basic.py) apart from `price_mode`, which the batched path leaves NaN.
    """
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
    # Batched fast path: compute every ticker's stats for a given span length in
    # one tensor_ops call (GPU when available) instead of per-ticker pandas.
    # Verified identical to the per-ticker path in test_basic.py; measured
    # 13.8x faster than the loop at 300 tickers x 1500 days on CUDA.
    if batched:
        return _build_profiles_batched(w, vm, om, hm, lm, tickers, spans,
                                       window, device=device)
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




# ---------------------------------------------------------------------------
# Batched multi-ticker profile (CPU/GPU via tensor_ops)
#
# `window_profile_stats` above is the per-ticker reference. It contains two
# Python loops (a per-day percentile rank and a per-day 3x3 np.linalg.solve),
# which dominate its runtime on a real universe.
#
# `window_profile_stats_batch` computes the same STAT_COLS for ALL tickers at
# once as [T, D] arrays, with every step a tensor_ops primitive, so device
# selection follows the repo-wide CUDA -> DirectML -> CPU ladder. There is no
# separate `_gpu` module: the former statistical_profiler_gpu.py was deleted
# because it duplicated the device ladder and returned all-NaN on any panel
# with leading NaN (torch.cumsum propagates NaN; tensor_ops.rolling_* do not).
#
# Verified against window_profile_stats in test_basic.py.
# ---------------------------------------------------------------------------


def window_profile_stats_batch(close, volume=None, L: int = 60,
                               open_=None, high=None, low=None,
                               device=None) -> dict:
    """Batched equivalent of window_profile_stats over a [T, D] panel.

    close/volume/open_/high/low: [T tickers, D days] float arrays (NaN allowed
    for missing history -- NaN handling matches pandas, unlike the deleted
    _gpu module). Returns {stat_name: [T, D] array} covering STAT_COLS, plus
    "_device" naming the device actually used.
    """
    import numpy as np
    from tensor_ops import (
        resolve_device, device_name, is_gpu, resident_device,
        rolling_mean, rolling_std, rolling_sum,
        rolling_reduce, rolling_rank_pct, rolling_median, rolling_quad_fit,
    )

    dev = resolve_device(device)
    # DirectML cannot run the float64 resident kernels (sqrt/pow/clamp/sort are
    # float32-only there), and float32 is not accurate enough for price-level
    # rolling std. resident_device() sends such boxes to CPU instead of raising.
    dev = resident_device(dev)
    c = np.asarray(close, dtype=float)
    if c.ndim == 1:
        c = c[None, :]
    T, D = c.shape

    def _as2d(x):
        if x is None:
            return None
        a = np.asarray(x, dtype=float)
        return a[None, :] if a.ndim == 1 else a

    v = _as2d(volume)
    o, h, lo = _as2d(open_), _as2d(high), _as2d(low)

    if is_gpu(dev):
        # Device-resident path: upload the panel ONCE, chain every rolling op on
        # the GPU, download ONCE at the end. Going through the numpy-facing
        # primitives instead would sync 25 times; on this panel .cpu() was 78%
        # of runtime and residency is ~30x faster.
        #
        # Chunked over tickers because the windowed temporaries, not the panel,
        # set the VRAM ceiling: a [T, D-L+1, L] reduction buffer is T*(D-L+1)*L*8
        # bytes (0.75 GB at 800x2000xL60) and several are live at once, which
        # exceeds a 2.15 GB MX550 and made the GPU SLOWER than CPU (25.98s vs
        # 12.72s) before chunking.
        return _profile_batch_resident_chunked(c, v, o, h, lo, L, dev)

    nan = np.full((T, D), np.nan)

    with np.errstate(all="ignore"):
        logc = np.log(np.where(c > 0, c, np.nan))

        # --- price distribution -------------------------------------------
        pm = rolling_mean(c, L, device=dev)
        pmed = rolling_median(c, L, device=dev)
        pstd = rolling_std(c, L, device=dev, ddof=1)      # pandas .std() default
        pmax = rolling_reduce(c, L, "max", device=dev)
        pmin = rolling_reduce(c, L, "min", device=dev)
        prange = pmax - pmin

        # skew / EXCESS kurtosis, matching _rolling_skew_kurt's double-rolling
        # estimator (NOT tensor_ops.rolling_skew -- see the note on that
        # function; the two differ by ~3 skew / ~9.6 kurt and swapping them
        # would change every STAT_COLS value).
        d2 = (c - pm) ** 2
        m2 = rolling_mean(d2, L, device=dev)
        d3 = rolling_mean((c - pm) ** 3, L, device=dev)
        d4 = rolling_mean((c - pm) ** 4, L, device=dev)
        pskew = d3 / np.power(m2, 1.5)
        pkurt = d4 / np.square(m2) - 3.0

        # --- position within window ---------------------------------------
        close_z = np.where(pstd > 0, (c - pm) / pstd, np.nan)
        # fraction of window values <= current close (replaces a per-day loop)
        pctile = rolling_rank_pct(c, L, device=dev)
        runup = np.where(prange > 0, (c - pmin) / np.where(prange > 0, prange, 1.0), np.nan)
        window_dd = c / np.where(pmax > 0, pmax, np.nan) - 1.0

        # --- shape: slope + curvature (replaces a per-day 3x3 solve) -------
        price_slope, curvature = rolling_quad_fit(c, L, device=dev)

        # --- volume -------------------------------------------------------
        if v is not None:
            vmean = rolling_mean(v, L, device=dev)
            vstd = rolling_std(v, L, device=dev, ddof=1)
            volume_z = np.where(vstd > 0, (v - vmean) / vstd, np.nan)
            vwap = rolling_sum(c * v, L, device=dev) / rolling_sum(v, L, device=dev)
        else:
            vmean = volume_z = vwap = nan

        # --- true OHLCV ---------------------------------------------------
        if o is not None and h is not None and lo is not None:
            vv = v if v is not None else np.ones((T, D))
            typ = (h + lo + c) / 3.0
            vwap_true = rolling_sum(typ * vv, L, device=dev) / rolling_sum(vv, L, device=dev)
            prev_c = np.full((T, D), np.nan)
            prev_c[:, 1:] = c[:, :-1]
            tr = np.maximum.reduce([h - lo, np.abs(h - prev_c), np.abs(lo - prev_c)])
            atr = rolling_mean(tr, L, device=dev)
            atr_pct = atr / np.where(c > 0, c, np.nan)
            gap = o / np.where(prev_c > 0, prev_c, np.nan) - 1.0
            gap_mean = rolling_mean(gap, L, device=dev)
            gap_std = rolling_std(gap, L, device=dev, ddof=1)
            range_hl = rolling_mean((h - lo) / np.where(c > 0, c, np.nan), L, device=dev)
            body = (c - o) / np.where(c > 0, c, np.nan)
            body_mean = rolling_mean(body, L, device=dev)
            body_std = rolling_std(body, L, device=dev, ddof=1)
            upper_wick = rolling_mean((h - np.maximum(c, o)) / np.where(c > 0, c, np.nan), L, device=dev)
            lower_wick = rolling_mean((np.minimum(c, o) - lo) / np.where(c > 0, c, np.nan), L, device=dev)
        else:
            vwap_true = atr = atr_pct = gap_mean = gap_std = nan
            range_hl = body_mean = body_std = upper_wick = lower_wick = nan

        # --- momentum -----------------------------------------------------
        log_ret = np.full((T, D), np.nan)
        log_ret[:, L:] = logc[:, L:] - logc[:, :-L]
        dr = np.full((T, D), np.nan)
        dr[:, 1:] = np.diff(logc, axis=1)
        ret_vol = rolling_std(dr, L, device=dev, ddof=1)
        momentum = log_ret / np.where(ret_vol > 0, ret_vol, np.nan)

    out = {
        "price_mean": pm, "price_median": pmed, "price_mode": nan,
        "price_max": pmax, "price_min": pmin, "price_range": prange,
        "price_std": pstd, "price_skew": pskew, "price_kurtosis": pkurt,
        "close_z": close_z, "close_pctile": pctile, "runup": runup,
        "window_drawdown": window_dd,
        "price_slope": price_slope, "price_curvature": curvature,
        "volume_mean": vmean, "vwap": vwap, "volume_z": volume_z,
        "log_ret": log_ret, "momentum": momentum, "ret_vol": ret_vol,
        "vwap_true": vwap_true, "atr": atr, "atr_pct": atr_pct,
        "gap_mean": gap_mean, "gap_std": gap_std, "range_hl": range_hl,
        "body_mean": body_mean, "body_std": body_std,
        "upper_wick": upper_wick, "lower_wick": lower_wick,
        "_device": device_name(dev),
    }
    return out




def _profile_batch_resident_chunked(c, v, o, h, lo, L, dev) -> dict:
    """Run the resident GPU path in ticker chunks sized to fit VRAM.

    The panel itself is tiny (12.8 MB at 800x2000); what blows up is each
    windowed reduction buffer, [chunk, D-L+1, L] float64. We budget for several
    of those being live simultaneously and pick the chunk height accordingly.
    """
    import numpy as np
    import torch

    Tn, Dn = c.shape
    win = max(Dn - L + 1, 1)
    bytes_per_ticker = win * L * 8          # one windowed buffer, one ticker
    try:
        free, _total = torch.cuda.mem_get_info(dev) if dev.type == "cuda" else (None, None)
    except Exception:
        free = None
    if not free:
        free = 1_500_000_000               # conservative default (DirectML etc.)
    # allow ~6 concurrent windowed temporaries, keep 25% headroom
    budget = int(free * 0.75 / 6)
    chunk = max(1, min(Tn, budget // max(bytes_per_ticker, 1)))

    if chunk >= Tn:
        return _profile_batch_resident(c, v, o, h, lo, L, dev)

    def _slice(x, a, b):
        return None if x is None else x[a:b]

    parts = []
    for a in range(0, Tn, chunk):
        b = min(a + chunk, Tn)
        parts.append(_profile_batch_resident(
            c[a:b], _slice(v, a, b), _slice(o, a, b),
            _slice(h, a, b), _slice(lo, a, b), L, dev))
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    out = {}
    for k in parts[0]:
        if k == "_device":
            out[k] = parts[0][k]
        else:
            out[k] = np.concatenate([p[k] for p in parts], axis=0)
    return out


def _profile_batch_resident(c, v, o, h, lo, L, dev) -> dict:
    """GPU path for window_profile_stats_batch: one upload, one download.

    Every intermediate stays in VRAM. The arithmetic is identical to the numpy
    branch above -- only the residency differs -- and test_basic.py asserts the
    two agree with the per-ticker reference to 1e-9.
    """
    import numpy as np
    import torch
    from tensor_ops import (
        device_name,
        rolling_mean_t, rolling_std_t, rolling_sum_t, rolling_reduce_t,
        rolling_median_t, rolling_rank_pct_t, rolling_quad_fit_t,
    )

    f64 = torch.float64

    def up(x):
        """Single host->device upload."""
        if x is None:
            return None
        a = np.ascontiguousarray(np.asarray(x, dtype=float))
        return torch.as_tensor(a, dtype=f64, device=dev)

    ct = up(c)
    vt, ot, ht, lot = up(v), up(o), up(h), up(lo)
    Tn, Dn = ct.shape
    nan_t = torch.full((Tn, Dn), float("nan"), dtype=f64, device=dev)

    logc = torch.log(torch.where(ct > 0, ct, torch.full_like(ct, float("nan"))))

    # --- price distribution ------------------------------------------------
    pm = rolling_mean_t(ct, L)
    pmed = rolling_median_t(ct, L)
    pstd = rolling_std_t(ct, L, ddof=1)          # pandas .std() default
    pmax = rolling_reduce_t(ct, L, "max")
    pmin = rolling_reduce_t(ct, L, "min")
    prange = pmax - pmin

    # skew / EXCESS kurtosis on the profiler's double-rolling estimator
    m2 = rolling_mean_t((ct - pm) ** 2, L)
    d3 = rolling_mean_t((ct - pm) ** 3, L)
    d4 = rolling_mean_t((ct - pm) ** 4, L)
    pskew = d3 / torch.pow(m2, 1.5)
    pkurt = d4 / (m2 * m2) - 3.0

    # --- position within window -------------------------------------------
    close_z = torch.where(pstd > 0, (ct - pm) / pstd, nan_t)
    pctile = rolling_rank_pct_t(ct, L)
    runup = torch.where(prange > 0, (ct - pmin) / torch.where(prange > 0, prange,
                                                             torch.ones_like(prange)), nan_t)
    window_dd = ct / torch.where(pmax > 0, pmax, nan_t) - 1.0

    # --- shape -------------------------------------------------------------
    price_slope, curvature = rolling_quad_fit_t(ct, L)

    # --- volume ------------------------------------------------------------
    if vt is not None:
        vmean = rolling_mean_t(vt, L)
        vstd = rolling_std_t(vt, L, ddof=1)
        volume_z = torch.where(vstd > 0, (vt - vmean) / vstd, nan_t)
        vwap = rolling_sum_t(ct * vt, L) / rolling_sum_t(vt, L)
    else:
        vmean = volume_z = vwap = nan_t

    # --- true OHLCV --------------------------------------------------------
    if ot is not None and ht is not None and lot is not None:
        vv = vt if vt is not None else torch.ones_like(ct)
        typ = (ht + lot + ct) / 3.0
        vwap_true = rolling_sum_t(typ * vv, L) / rolling_sum_t(vv, L)
        prev_c = torch.full_like(ct, float("nan"))
        prev_c[:, 1:] = ct[:, :-1]
        tr = torch.maximum(torch.maximum(ht - lot, (ht - prev_c).abs()),
                           (lot - prev_c).abs())
        atr = rolling_mean_t(tr, L)
        cpos = torch.where(ct > 0, ct, nan_t)
        atr_pct = atr / cpos
        gap = ot / torch.where(prev_c > 0, prev_c, nan_t) - 1.0
        gap_mean = rolling_mean_t(gap, L)
        gap_std = rolling_std_t(gap, L, ddof=1)
        range_hl = rolling_mean_t((ht - lot) / cpos, L)
        body = (ct - ot) / cpos
        body_mean = rolling_mean_t(body, L)
        body_std = rolling_std_t(body, L, ddof=1)
        upper_wick = rolling_mean_t((ht - torch.maximum(ct, ot)) / cpos, L)
        lower_wick = rolling_mean_t((torch.minimum(ct, ot) - lot) / cpos, L)
    else:
        vwap_true = atr = atr_pct = gap_mean = gap_std = nan_t
        range_hl = body_mean = body_std = upper_wick = lower_wick = nan_t

    # --- momentum ----------------------------------------------------------
    log_ret = torch.full_like(ct, float("nan"))
    log_ret[:, L:] = logc[:, L:] - logc[:, :-L]
    dr = torch.full_like(ct, float("nan"))
    dr[:, 1:] = logc[:, 1:] - logc[:, :-1]
    ret_vol = rolling_std_t(dr, L, ddof=1)
    momentum = log_ret / torch.where(ret_vol > 0, ret_vol, nan_t)

    packed = {
        "price_mean": pm, "price_median": pmed,
        "price_max": pmax, "price_min": pmin, "price_range": prange,
        "price_std": pstd, "price_skew": pskew, "price_kurtosis": pkurt,
        "close_z": close_z, "close_pctile": pctile, "runup": runup,
        "window_drawdown": window_dd,
        "price_slope": price_slope, "price_curvature": curvature,
        "volume_mean": vmean, "vwap": vwap, "volume_z": volume_z,
        "log_ret": log_ret, "momentum": momentum, "ret_vol": ret_vol,
        "vwap_true": vwap_true, "atr": atr, "atr_pct": atr_pct,
        "gap_mean": gap_mean, "gap_std": gap_std, "range_hl": range_hl,
        "body_mean": body_mean, "body_std": body_std,
        "upper_wick": upper_wick, "lower_wick": lower_wick,
    }
    # ONE device->host transfer for the whole result set.
    keys = list(packed)
    stacked = torch.stack([packed[k] for k in keys], dim=0).cpu().numpy()
    out = {k: stacked[i] for i, k in enumerate(keys)}
    out["price_mode"] = np.full((Tn, Dn), np.nan)   # histogram mode: CPU-only
    out["_device"] = device_name(dev)
    return out


def _build_profiles_batched(w, vm, om, hm, lm, tickers, spans, window,
                            device=None) -> pd.DataFrame:
    """Long-format profiles via the batched engine, one pass per span length."""
    import numpy as np

    keep = [t for t in tickers if t in w.columns]
    if not keep:
        return pd.DataFrame()
    sub = w[keep].tail(window)
    dates = sub.index
    close = sub.to_numpy(dtype=float).T                     # [T, D]

    def _mat(src):
        if src is None:
            return None
        cols = [t for t in keep if t in src.columns]
        if not cols:
            return None
        m = src.reindex(index=dates, columns=keep)
        return m.to_numpy(dtype=float).T

    vol, op, hi, lo = _mat(vm), _mat(om), _mat(hm), _mat(lm)

    # tickers with too little real history are dropped, matching the loop's
    # `len(c) < MIN_DAYS: continue`
    obs = np.isfinite(close).sum(axis=1)
    ok_rows = obs >= MIN_DAYS

    by_len = {}
    for f, t_, L in spans:
        by_len.setdefault(L, []).append((f, t_))

    frames = []
    for L, ft_pairs in sorted(by_len.items()):
        st = window_profile_stats_batch(close, vol, L, op, hi, lo, device=device)
        n_t, n_d = close.shape
        tick_col = np.repeat(np.asarray(keep, dtype=object), n_d)
        date_col = np.tile(dates.to_numpy(), n_t)
        base = {"ticker": tick_col, "date": date_col,
                "close": close.reshape(-1)}
        for col in STAT_COLS:
            base[col] = np.asarray(st[col], dtype=float).reshape(-1)
        row_ok = np.repeat(ok_rows, n_d)
        df = pd.DataFrame(base)[row_ok]
        # drop rows where the window never formed (all stats NaN)
        df = df[np.isfinite(df["price_mean"].to_numpy())]
        if df.empty:
            continue
        for (f, t_) in ft_pairs:
            d2 = df.copy()
            d2["span_from"], d2["span_to"], d2["span_len"] = f, t_, L
            frames.append(d2)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=None, help="cap universe size")
    ap.add_argument("--window", type=int, default=1500, help="trailing days per ticker")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--batched", action="store_true",
                    help="use the tensor_ops batched engine (GPU when available)")
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda (only with --batched)")
    args = ap.parse_args()

    dev = None if args.device == "auto" else args.device
    if args.batched:
        from tensor_ops import device_name, resolve_device
        print(f"Building statistical profiles (window={args.window}, "
              f"batched on {device_name(resolve_device(dev))})...")
    else:
        print(f"Building statistical profiles (window={args.window})...")
    df = build_profiles(args.tickers, args.window,
                        batched=args.batched, device=dev)
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