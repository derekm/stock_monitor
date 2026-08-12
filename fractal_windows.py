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

    Default configs: [(30,3), (10,3)] → 90-day view + 30-day view.
    Returns: {f"{a}x{b}": {"signal": DataFrame, "consensus": DataFrame, "best": DataFrame}}
    """
    if configs is None:
        configs = [(30, 3), (10, 3)]
    out = {}
    for a, b in configs:
        fdf = fractal_signal_vec(close, a, b)
        cons = fractal_consensus(fdf)
        best = best_span_wins(fdf)
        out[f"{a}x{b}"] = {"signal": fdf, "consensus": cons, "best": best}
    return out


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
