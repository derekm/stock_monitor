#!/usr/bin/env python3
"""fractal_windows.py — fractal sliding-window momentum (US20120253946A1, FIGS 28-29).

Implements the fractal-of-sliding-windows scheme from the patent I'm an inventor
on, applied to judging momentum on a rolling forward basis.

Scheme (FIGS 26A/28/29):
  - base span `a` and repetitions `b` tile the total range [0, a*b].
  - spans_generator(a, b) emits ALL window-aligned (x, y) pairs with
    x, y in {0, a, 2a, ..., a*b}, x < y. For (30, 3):
      (0,30) (0,60) (0,90) (30,60) (30,90) (60,90)
    i.e. every length that's a multiple of the base, at every aligned start —
    the "fractal" of the range.
  - statistical_profiles_generator(a, b, c, d) slides each fractal span forward
    over time offsets `past` in [c, d] (FIG 29), computing a statistical profile
    (here: momentum return, mean, vol, slope) on each window
    [t + from, t + to] where t = current rolling offset.

Why: a single momentum window (e.g. 12m) is arbitrary. The fractal scheme judges
the same range at EVERY aligned granularity simultaneously, so a breakout shows
up consistently across small and large windows (self-similarity) — a stronger
signal than any one window alone. The sliding `past` offset makes it rolling
forward, matching the patent's day-by-day re-ranking.

All functions are pure (operate on a DatetimeIndexed log-return / close series).
No data I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── FIG 28: fractal span generator ───────────────────────────────────────
def spans_generator(a: int, b: int) -> list[tuple[int, int]]:
    """Emit all fractal spans (x, y) for base `a`, repetitions `b`.

    Faithful port of FIG 28's PL/pgSQL:
      lim = a*b; x=0; i=1
      loop: if x==lim exit; y=a*i; return (x,y);
            if y==lim: x+=a; i=x/a;  i+=1
    Verified: (30,3) -> (0,30)(0,60)(0,90)(30,60)(30,90)(60,90).
    """
    lim = a * b
    x, i = 0, 1
    spans: list[tuple[int, int]] = []
    while True:
        if x == lim:
            break
        y = a * i
        spans.append((x, y))
        if y == lim:
            x = x + a
            i = x // a
        i = i + 1
    # dedupe / sort by (from, to)
    return sorted(set(spans))


# ── FIG 29: sliding-window statistical profiles ─────────────────────────
def window_profiles(log_ret: pd.Series, span_from: int, span_to: int,
                    past: int) -> dict:
    """Statistical profile (momentum-focused) of one fractal window.

    Window in DAYS: [past + from, past + to] (patent uses day offsets; here we
    map 1 unit = 1 trading day for simplicity, so `past` steps by 1 day).

    Returns momentum return over the window, mean daily ret, vol, and slope
    (regression of cumulative return on time). All from log returns.
    """
    lo = past + span_from
    hi = past + span_to
    w = log_ret.iloc[lo:hi]
    if len(w) < 5:
        return {"n": 0, "ret": np.nan, "mean": np.nan, "vol": np.nan, "slope": np.nan}
    cum = w.cumsum()
    x = np.arange(len(w), dtype=float)
    slope = np.polyfit(x, cum.values, 1)[0] if len(w) > 2 else np.nan
    return {
        "n": int(len(w)),
        "ret": float(w.sum()),
        "mean": float(w.mean()),
        "vol": float(w.std()),
        "slope": float(slope),
    }


def fractal_signal(close: pd.Series, a: int, b: int,
                   past_start: int = 0, past_end: int | None = None) -> pd.DataFrame:
    """Compute a momentum signal for EVERY fractal span, sliding forward.

    close: DatetimeIndexed price series (forward-filled). Returns a long
    DataFrame: one row per (rolling date, span), with window stats + whether the
    span is in an uptrend (ret>0 and slope>0) and momentum magnitude (ret/vol).

    Backward-looking framing: at rolling date index `i`, each span (x,y) covers
    the TRAILING window [i-(y-x), i] (i.e. past = i - y, so past+y = i). All
    fractal spans END at the current date, giving the full multi-granularity
    decomposition of the trailing a*b-day range.
    """
    log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(log_ret)
    spans = spans_generator(a, b)
    if past_end is None:
        past_end = n - 1  # last date index (windows need only past data)
    past_end = max(past_start, min(past_end, n - 1))

    rows = []
    for i in range(max(past_start, a * b), past_end + 1):
        for (f0, t0) in spans:
            # trailing window ending at i: [i-(t0-f0), i]
            f = i - (t0 - f0)
            w = log_ret.iloc[f:i]
            if len(w) < 5:
                continue
            cum = w.cumsum()
            xv = np.arange(len(w), dtype=float)
            slope = np.polyfit(xv, cum.values, 1)[0] if len(w) > 2 else np.nan
            p = {
                "n": int(len(w)),
                "ret": float(w.sum()),
                "mean": float(w.mean()),
                "vol": float(w.std()),
                "slope": float(slope),
            }
            m = p["ret"] / p["vol"] if p["vol"] and p["vol"] > 0 else np.nan
            rows.append({
                "past": i,
                "date": log_ret.index[i],
                "span_from": f0,
                "span_to": t0,
                "span_len": t0 - f0,
                "ret": p["ret"],
                "mean": p["mean"],
                "vol": p["vol"],
                "slope": p["slope"],
                "momentum": m,
                "uptrend": bool(p["ret"] > 0 and (pd.notna(p["slope"]) and p["slope"] > 0)),
            })
    return pd.DataFrame(rows)


# ── agreement / consensus across fractal spans ──────────────────────────
def fractal_consensus(df: pd.DataFrame, span_len: int | None = None) -> pd.DataFrame:
    """Per `date` (rolling offset), the fraction of fractal spans in an uptrend
    and the mean risk-adjusted momentum. This is the self-similarity signal: a
    breakout shows up across MANY spans at once.

    If span_len given, only consider spans of exactly that length (isolate the
    base granularity). Otherwise all spans.

    Uses polars group_by when available (the repo's established fast path for
    long-frame aggregations — see peer_analytics / crisis_correlation), falling
    back to pandas. Output schema is identical either way.
    """
    d = df if span_len is None else df[df["span_len"] == span_len]
    if d.empty:
        return pd.DataFrame()
    try:
        import polars as pl
        pdf = pl.from_pandas(d)
        out = (pdf.group_by("date").agg(
            n_spans=pl.col("momentum").count(),
            frac_uptrend=pl.col("uptrend").mean(),
            mean_momentum=pl.col("momentum").mean(),
            mean_ret=pl.col("ret").mean(),
            mean_slope=pl.col("slope").mean(),
        ).sort("date").to_pandas())
        return out.set_index("date")
    except Exception:  # noqa: BLE001  (polars unavailable -> pandas fallback)
        g = d.groupby("date").agg(
            n_spans=("momentum", "count"),
            frac_uptrend=("uptrend", "mean"),
            mean_momentum=("momentum", "mean"),
            mean_ret=("ret", "mean"),
            mean_slope=("slope", "mean"),
        )
        return g


def breakout_score(consensus: pd.DataFrame,
                   uptrend_thresh: float = 0.6, momentum_thresh: float = 0.5) -> pd.Series:
    """A 0/1 breakout signal: the fractal range is 'freshly exploding' when the
    majority of spans are in an uptrend AND risk-adjusted momentum is strong."""
    return ((consensus["frac_uptrend"] >= uptrend_thresh) &
            (consensus["mean_momentum"] >= momentum_thresh)).astype(int)


def best_span_wins(df: pd.DataFrame, metric: str = "momentum") -> pd.DataFrame:
    """Per `date`, the BEST fractal span — the patent's ranking/selection view.

    Where `fractal_consensus` averages across all spans (self-similarity), this
    picks the single span with the strongest `metric` at each date (default:
    risk-adjusted momentum, matching the patent's winner-take-all re-ranking).
    A breakout is 'confirmed' by the best span if it's in an uptrend AND its
    momentum is positive — a looser, earlier trigger than consensus (which needs
    a majority of spans). Comparing the two answers "is this a broad multi-granular
    breakout, or a narrow single-window pop?"

    Returns per-date: winning span (from, to, len), its ret/slope/momentum/
    uptrend, plus `confirmed` (best span uptrend AND momentum > 0).
    """
    d = df[df[metric].notna()]
    if d.empty:
        return pd.DataFrame()
    try:
        import polars as pl
        pdf = pl.from_pandas(d)
        best = (pdf.sort("date", metric, descending=[False, True])
                .group_by("date", maintain_order=True)
                .first())
        out = best.select([
            "date",
            pl.col("span_from").alias("best_span_from"),
            pl.col("span_to").alias("best_span_to"),
            pl.col("span_len").alias("best_span_len"),
            pl.col("ret").alias("best_ret"),
            pl.col("slope").alias("best_slope"),
            pl.col("momentum").alias("best_momentum"),
            pl.col("uptrend").alias("best_uptrend"),
        ]).sort("date").to_pandas()
    except Exception:  # noqa: BLE001  (polars unavailable -> pandas)
        g = d.sort_values(["date", metric], ascending=[True, False])
        g = g.groupby("date", sort=True).first()
        out = g[["span_from", "span_to", "span_len", "ret", "slope", "momentum", "uptrend"]]
        out.columns = ["best_span_from", "best_span_to", "best_span_len",
                       "best_ret", "best_slope", "best_momentum", "best_uptrend"]
        out = out.reset_index()
    out = out.set_index("date")
    out["confirmed"] = (out["best_uptrend"].astype(bool) & (out["best_momentum"] > 0)).astype(int)
    return out


def fractal_multi_view(close: pd.Series, configs: list[tuple[int, int]] = None
                       ) -> dict[str, dict]:
    """Run fractal signal + consensus + best_span for multiple (a,b) configs.

    Default configs span the granularity ladder — each `a*b` is the full window:
      [(5,3), (10,3), (15,3), (30,3)] -> 15d, 30d, 45d, 90d views.
    Returns: {f"{a}x{b}": {"signal": DataFrame, "consensus": DataFrame, "best": DataFrame}}
    """
    if configs is None:
        configs = [(5, 3), (10, 3), (15, 3), (30, 3)]
    out = {}
    for a, b in configs:
        fdf = fractal_signal_vec(close, a, b)
        cons = fractal_consensus(fdf)
        best = best_span_wins(fdf)
        out[f"{a}x{b}"] = {"signal": fdf, "consensus": cons, "best": best}
    return out


def fractal_posture(views: dict[str, dict],
                    uptrend_thresh: float = 0.6) -> dict:
    """Classify a ticker's fractal posture from multi-view results.

    `views` is the output of `fractal_multi_view`. Classifies whether a breakout
    is BROAD (self-similar — best span confirmed AND consensus high in MOST
    views) or NARROW (only some views confirm), plus a freshness/aging read:

    posture:
      BROAD   — best span confirmed + consensus >= thresh in >=2/3 of views
      MIXED   — best span confirmed in most views but consensus below thresh
                (trend intact, breadth thinning)
      NARROW  — best span confirmed in <half of views (single-window pop, not
                self-similar)
      WEAK    — no view confirms (not a breakout)

    Returns dict: posture, n_views, n_confirmed, n_broad, trend, freshness.
    `freshness`: 're-accelerating' (majority of views' consensus rising MoM) /
                 'stalling' (falling) / 'steady'.
    """
    if not views:
        return {"posture": "WEAK", "n_views": 0, "n_confirmed": 0,
                "n_broad": 0, "trend": "flat", "freshness": "steady"}
    n = len(views)
    n_confirmed = 0
    n_broad = 0
    trend_up = 0
    trend_dn = 0
    for key, v in views.items():
        cons = v["consensus"]
        best = v["best"]
        if not len(cons) or not len(best):
            continue
        conf = int(best["confirmed"].iloc[-1])
        frac = float(cons["frac_uptrend"].iloc[-1])
        if conf == 1:
            n_confirmed += 1
        if conf == 1 and frac >= uptrend_thresh:
            n_broad += 1
        # MoM trend of consensus fraction
        f = cons["frac_uptrend"]
        if len(f) >= 2:
            prev = float(f.iloc[-2]) if f.index[-1] != f.index[-2] else float(f.iloc[max(0, -3)])
            cur = float(f.iloc[-1])
            if cur > prev + 1e-6:
                trend_up += 1
            elif cur < prev - 1e-6:
                trend_dn += 1
    if n_confirmed == 0:
        posture = "WEAK"
    elif 2 * n_broad > n:        # strict majority of views fully broad (self-similar)
        posture = "BROAD"
    elif 2 * n_confirmed > n:    # best span confirmed in most views but breadth thinning
        posture = "MIXED"
    else:
        posture = "NARROW"
    if trend_up > trend_dn:
        freshness = "re-accelerating"
    elif trend_dn > trend_up:
        freshness = "stalling"
    else:
        freshness = "steady"
    return {
        "posture": posture, "n_views": n,
        "n_confirmed": n_confirmed, "n_broad": n_broad,
        "trend": "rising" if trend_up > trend_dn else ("falling" if trend_dn > trend_up else "flat"),
        "freshness": freshness,
    }


def momentum_stack(views: dict[str, dict]) -> dict:
    """Chain-of-spans measure: do consecutive DIFFERING-LENGTH spans build momentum?

    The granularity ladder [(5,3)=15d, (10,3)=30d, (15,3)=45d, (30,3)=90d] gives
    windows of increasing length. A strong ride case requires not just each view
    confirming, but a MONOTONIC build: the longest window shows the highest
    (or positive) momentum, and confirmation holds across consecutive lengths.

    A sustained breakout looks like a stack: 15d up, 30d up, 45d up, 90d up —
    the shorter windows confirm the impulse and the longer windows confirm the
    trend has breadth. A broken stack (e.g. 15d down, 30d up, 45d up, 90d up)
    is a short-term pullback inside a longer uptrend.

    Returns:
      stack_depth  — length of the longest run of views (short->long) where the
                     best span is confirmed (0..n_views)
      full_stack   — True if ALL views confirmed (strongest ride case)
      base_confirmed, mid_confirmed, top_confirmed — 15d, 45d, 90d flags
      monotonic    — True if confirmation is monotonic across the ladder (once a
                     view fails, no later view re-confirms — a clean stack)
      stack_mom    — mean best-span momentum across confirmed views
    """
    if not views:
        return {"stack_depth": 0, "full_stack": False, "base_confirmed": False,
                "mid_confirmed": False, "top_confirmed": False,
                "monotonic": True, "stack_mom": 0.0}
    # order views short->long by full-window length (a*b)
    ordered = sorted(views.items(), key=lambda kv: int(kv[0].split("x")[0]) * int(kv[0].split("x")[1]))
    flags = []
    moms = []
    for key, v in ordered:
        best = v["best"]
        conf = int(best["confirmed"].iloc[-1]) if len(best) else 0
        mom = float(best["best_momentum"].iloc[-1]) if len(best) else 0.0
        flags.append(bool(conf))
        moms.append(mom)
    # longest run of True from the SHORT end
    depth = 0
    for f in flags:
        if f:
            depth += 1
        else:
            break
    full = all(flags)
    # monotonic: once a False appears, no True after it
    mono = True
    seen_false = False
    for f in flags:
        if not f:
            seen_false = True
        elif seen_false:
            mono = False
    n = len(ordered)
    return {
        "stack_depth": depth,
        "full_stack": full,
        "base_confirmed": flags[0] if n else False,
        "mid_confirmed": flags[n // 2] if n else False,
        "top_confirmed": flags[-1] if n else False,
        "monotonic": mono,
        "stack_mom": float(np.mean([m for m, f in zip(moms, flags) if f])) if any(flags) else 0.0,
    }


def momentum_stack_series(views: dict[str, dict]) -> pd.DataFrame:
    """Historical per-date stack_depth — the full chain-of-spans time series.

    For every date in the views, counts how many consecutive views (short->long
    granularity ladder) have their best span confirmed. Returns a DataFrame
    indexed by date with `stack_depth` (0..n_views) and `full_stack` (bool).
    Used by the ride backtest to apply the quality gate / dual exit over time.
    """
    if not views:
        return pd.DataFrame(columns=["stack_depth", "full_stack"])
    # align all best frames on a common date index
    dates = None
    series = {}
    ordered = sorted(views.items(), key=lambda kv: int(kv[0].split("x")[0]) * int(kv[0].split("x")[1]))
    for key, v in ordered:
        best = v["best"]
        if best is None or len(best) == 0:
            continue
        conf = best["confirmed"].astype(int)
        series[key] = conf
        dates = conf.index if dates is None else dates.union(conf.index)
    if not series:
        return pd.DataFrame(columns=["stack_depth", "full_stack"])
    frame = pd.DataFrame({k: s.reindex(dates) for k, s in series.items()})
    frame = frame.sort_index().ffill().fillna(0).astype(int)
    # stack_depth = longest run of 1s from the SHORT end across the ladder
    keys = [k for k in series]
    depth = pd.Series(0, index=frame.index, dtype=int)
    for i, k in enumerate(keys):
        run = frame[keys[: i + 1]].all(axis=1)   # first i+1 views all confirmed
        depth = depth.mask(run, i + 1)
    return pd.DataFrame({"stack_depth": depth, "full_stack": (depth == len(keys)).astype(int)})


# ── vectorized fractal signal (no per-day polyfit loops) ─────────────────
def fractal_signal_vec(close: pd.Series, a: int, b: int) -> pd.DataFrame:
    """Vectorized fractal signal using rolling-window momentum.

    For each span length L in the fractal (a, 2a, ..., a*b), computes on EVERY
    date the trailing L-day log return and the linear slope of log-price over
    the trailing L-day window — both via rolling sums (O(n), no per-day polyfit).
    Returns a long DataFrame: one row per (date, span), with ret/slope/uptrend.

    Slope of y=log-price vs x=0..L-1 over a trailing window, closed-form:
        slope = [L*Sxy - Sx*Sy] / [L*Sxx - Sx^2]
        Sx = L(L-1)/2, Sxx = L(L-1)(2L-1)/6   (constant)
        Sy = rolling_sum(y) over window
        Sxy = rolling_sum(k*y) - (t-L+1)*rolling_sum(y)  [k = global idx]
    """
    logp = np.log(close).replace([np.inf, -np.inf], np.nan)
    idx = np.arange(len(logp), dtype=float)
    k_y = pd.Series(idx * logp.values, index=logp.index)  # k * y
    spans = spans_generator(a, b)
    lengths = sorted({t - f for f, t in spans})

    rows = []
    for L in lengths:
        if len(logp) < L + 1:
            continue
        # trailing L-day log return: ret = logp[t] - logp[t-L]
        ret = logp - logp.shift(L)
        # rolling sums over window of size L (window = last L rows, ends at t)
        sy = logp.rolling(L).sum()             # sum of y in window
        sky = k_y.rolling(L).sum()             # sum of k*y in window
        # window start index = t - L + 1
        start = idx - (L - 1)
        sxy = sky - start * sy                 # sum (k-(t-L+1))*y
        sx = L * (L - 1) / 2.0
        sxx = L * (L - 1) * (2 * L - 1) / 6.0
        denom = L * sxx - sx * sx
        slope = (L * sxy - sx * sy) / denom if denom != 0 else np.nan
        vol = logp.diff().rolling(L).std()     # daily-ret std over window
        # attach span (f,t) — all spans of this length share the same series
        for (f, t) in spans:
            if t - f != L:
                continue
            m = ret / vol.replace(0, np.nan)
            rows.append(pd.DataFrame({
                "date": logp.index,
                "span_len": L,
                "span_from": f,
                "span_to": t,
                "ret": ret.values,
                "slope": slope.values,
                "momentum": m.values,
                "uptrend": ((ret > 0) & (slope > 0)).values,
            }))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out


# ---------------------------------------------------------------------------
# Batched (multi-ticker) fractal engine
#
# Folded in from the former `fractal_windows_gpu.py` (deleted 2026-08). There is
# no separate `_gpu` module any more: device selection lives in `tensor_ops`, so
# a parallel "gpu" file only duplicated the CPU/GPU ladder and drifted from it.
# `fractal_batch` works on any device -- pass device="cpu" to force CPU.
#
# Two numeric bugs from the old file are fixed here:
#   1. It ran in float32. The rolling-variance identity (sum(x^2) - sum(x)^2/L)
#      subtracts two large nearly-equal numbers, so float32 produced errors of
#      O(100) on price-level input. Now float64.
#   2. It used raw torch.cumsum, which PROPAGATES NaN -- one NaN poisoned every
#      later value in the row. tensor_ops.rolling_sum treats NaN as 0 and masks
#      by observed count, matching pandas/polars.
# ---------------------------------------------------------------------------


def fractal_batch(wide_logp: np.ndarray, a: int = 30, b: int = 3,
                  device=None) -> dict:
    """wide_logp: [T tickers, D days] log-prices. Returns
    {(from, to): {ret, slope, momentum, uptrend, vol} as [T, D] tensors}.

    One batched pass over every ticker; the span tuples are integer loop
    indices, so no host<->device round-trips happen inside the loop.
    """
    import torch
    from tensor_ops import resolve_device, rolling_sum, rolling_std

    dev = resolve_device(device)
    T, D = wide_logp.shape
    spans = spans_generator(a, b)
    lengths = sorted({t - f for f, t in spans})

    # float64: see note above -- float32 breaks the variance identity.
    logp_np = np.asarray(wide_logp, dtype=float)
    logp = torch.as_tensor(logp_np, dtype=torch.float64, device=dev)
    idx_np = np.arange(D, dtype=float)
    idx = torch.as_tensor(idx_np, dtype=torch.float64, device=dev)

    # daily log returns, NaN on the first column (no prior observation)
    dr_np = np.full_like(logp_np, np.nan)
    dr_np[:, 1:] = np.diff(logp_np, axis=1)

    result = {}
    for L in lengths:
        # OLS slope of logp against the within-window index, via rolling sums.
        sy = rolling_sum(logp_np, L, device=dev)
        sky = rolling_sum(idx_np * logp_np, L, device=dev)
        start = idx_np - (L - 1)
        sxy = sky - start * sy
        sx = L * (L - 1) / 2.0
        sxx = L * (L - 1) * (2 * L - 1) / 6.0
        denom = L * sxx - sx * sx
        slope_np = (L * sxy - sx * sy) / denom

        # trailing L-day log return = logp[t] - logp[t-L]
        ret_np = np.full_like(logp_np, np.nan)
        ret_np[:, L:] = logp_np[:, L:] - logp_np[:, :-L]

        # rolling vol of daily returns (ddof=1, matching the original intent)
        vol_np = rolling_std(dr_np, L, device=dev, ddof=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            mom_np = ret_np / np.clip(vol_np, 1e-9, None)
        up_np = (ret_np > 0) & (slope_np > 0)

        ret_t = torch.as_tensor(ret_np, device=dev)
        slope_t = torch.as_tensor(slope_np, device=dev)
        mom_t = torch.as_tensor(mom_np, device=dev)
        vol_t = torch.as_tensor(vol_np, device=dev)
        up_t = torch.as_tensor(up_np, device=dev)

        for (f, t) in spans:
            if t - f != L:
                continue
            result[(f, t)] = {
                "ret": ret_t, "slope": slope_t, "momentum": mom_t,
                "uptrend": up_t, "vol": vol_t,
            }
    return result


def fractal_consensus_batch(res: dict, T: int, D: int, device=None) -> dict:
    """Consensus across all fractal spans: NaN-safe mean of each stat.

    The span dimension is a fixed known axis, so this is a stack + masked mean
    over dim 0 rather than a groupby. Returns [T, D] tensors.
    """
    import torch
    from tensor_ops import resolve_device

    dev = resolve_device(device)
    spans = list(res.keys())
    if not spans:
        return {}

    def nanmean(key, cast_float=False):
        x = torch.stack([res[s][key].to(torch.float64) for s in spans], 0)
        mask = torch.isfinite(x)
        s = torch.where(mask, x, torch.zeros_like(x)).sum(0)
        n = mask.sum(0).clamp(min=1)
        return s / n

    return {
        "frac_uptrend": nanmean("uptrend"),
        "mean_momentum": nanmean("momentum"),
        "mean_ret": nanmean("ret"),
        "mean_slope": nanmean("slope"),
        "n_spans": torch.full((T, D), float(len(spans)), device=dev,
                              dtype=torch.float64),
    }


def gpu_available() -> bool:
    """True when any accelerator is usable (CUDA or DirectML) -- see tensor_ops."""
    from tensor_ops import gpu_available as _ga
    return _ga()
