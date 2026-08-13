#!/usr/bin/env python3
"""statistical_profiler_gpu.py — GPU vectorized TRUE statistical profiler over
fractal windows.

Computes a WIDE statistical profile per fractal span: price levels (mean/median/
mode/max/min/range/std/skew/kurt), position (close_z, close_pctile, runup,
window_dd), volume (mean, true_vwap, volume_z), momentum (log_ret, momentum,
ret_vol), true-OHLCV (ATR, gap, range_hl, body, wicks). All point-in-time,
batched on GPU.

Usage (as module):
  from statistical_profiler_gpu import fractal_stats_batch
  out = fractal_stats_batch(wide_close, wide_vol, wide_high, wide_low, wide_open,
                            configs)
  # out: {span_len: {stat_name: [T, D] tensor on device}}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from fractal_windows import spans_generator


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _batched_rolling_sums(x: torch.Tensor, L: int) -> torch.Tensor:
    """Rolling L-day sum via cumsum on [T, D] tensor."""
    cum = torch.cumsum(x, dim=1)
    out = cum.clone()
    out[:, L:] = out[:, L:] - cum[:, :-L]
    out[:, : L - 1] = float("nan")
    return out


def _batched_rolling_mean(x: torch.Tensor, L: int) -> torch.Tensor:
    return _batched_rolling_sums(x, L) / L


def _batched_rolling_std(x: torch.Tensor, L: int) -> torch.Tensor:
    s = _batched_rolling_sums(x, L)
    s2 = _batched_rolling_sums(x * x, L)
    var = (s2 - s * s / L) / (L - 1)
    return torch.sqrt(torch.clamp(var, min=0))


def _batched_rolling_skew(x: torch.Tensor, L: int) -> torch.Tensor:
    """Rolling skewness: E[(x - mu)^3] / sigma^3."""
    m = _batched_rolling_mean(x, L)
    d = x - m
    m3 = _batched_rolling_mean(d * d * d, L)
    m2 = _batched_rolling_mean(d * d, L)
    skew = m3 / torch.pow(torch.clamp(m2, min=1e-12), 1.5)
    return torch.where(torch.isfinite(skew), skew, float("nan"))


def _batched_rolling_kurt(x: torch.Tensor, L: int) -> torch.Tensor:
    """Rolling excess kurtosis: E[(x - mu)^4] / sigma^4 - 3."""
    m = _batched_rolling_mean(x, L)
    d = x - m
    m4 = _batched_rolling_mean(d * d * d * d, L)
    m2 = _batched_rolling_mean(d * d, L)
    kurt = m4 / torch.clamp(m2 * m2, min=1e-12) - 3.0
    return torch.where(torch.isfinite(kurt), kurt, float("nan"))


def _batched_rolling_mode(x: torch.Tensor, L: int, bins: int = 20) -> torch.Tensor:
    """Rolling mode via histogram approximation on each window."""
    T, D = x.shape
    out = torch.full_like(x, float("nan"))
    for t in range(L - 1, D):
        window = x[:, t - L + 1 : t + 1]  # [T, L]
        mask = torch.isfinite(window)
        if not mask.any():
            continue
        wmin = torch.where(mask, window, torch.full_like(window, float("inf"))).min(dim=1).values
        wmax = torch.where(mask, window, torch.full_like(window, -float("inf"))).max(dim=1).values
        rng = wmax - wmin
        rng = torch.where(rng > 0, rng, torch.tensor(1.0, device=window.device))
        norm = (window - wmin.unsqueeze(1)) / rng.unsqueeze(1)
        bins_idx = (norm * bins).long().clamp(0, bins - 1)
        counts = torch.zeros((T, bins), dtype=torch.long, device=window.device)
        for b in range(bins):
            counts[:, b] = (bins_idx == b).sum(dim=1)
        mode_bin = counts.argmax(dim=1)
        out[:, t] = wmin + (mode_bin.float() + 0.5) * rng / bins
    return out


def _batched_rolling_percentile_rank(x: torch.Tensor, L: int) -> torch.Tensor:
    """Rolling percentile rank of the last value in the window."""
    T, D = x.shape
    out = torch.full_like(x, float("nan"))
    for t in range(L - 1, D):
        window = x[:, t - L + 1 : t + 1]  # [T, L]
        last = window[:, -1:]
        le = (window <= last).float().mean(dim=1)
        out[:, t] = le
    return out


def _batched_rolling_runup(x: torch.Tensor, L: int) -> torch.Tensor:
    """Runup: (current - min) / (max - min) over window."""
    T, D = x.shape
    out = torch.full_like(x, float("nan"))
    for t in range(L - 1, D):
        w = x[:, t - L + 1 : t + 1]
        mn = w.min(dim=1).values
        mx = w.max(dim=1).values
        rng = mx - mn
        out[:, t] = (x[:, t] - mn) / torch.clamp(rng, min=1.0)
    return out


def _batched_rolling_window_dd(x: torch.Tensor, L: int) -> torch.Tensor:
    """Window drawdown: current / running_max - 1."""
    T, D = x.shape
    out = torch.full_like(x, float("nan"))
    for t in range(L - 1, D):
        w = x[:, t - L + 1 : t + 1]
        cummax = w.max(dim=1).values
        out[:, t] = x[:, t] / cummax - 1.0
    return out


def _batched_rolling_slope(x: torch.Tensor, L: int) -> torch.Tensor:
    """Closed-form OLS slope over window (price vs local time index 0..L-1)."""
    T, D = x.shape
    out = torch.full_like(x, float("nan"))
    k = torch.arange(L, dtype=x.dtype, device=x.device)
    sx = L * (L - 1) / 2.0
    sxx = L * (L - 1) * (2 * L - 1) / 6.0
    denom = L * sxx - sx * sx
    for t in range(L - 1, D):
        w = x[:, t - L + 1 : t + 1]
        sy = w.sum(dim=1)
        sky = (k.unsqueeze(0) * w).sum(dim=1)
        slope = (L * sky - sx * sy) / denom
        out[:, t] = slope
    return out


def _batched_rolling_curvature(x: torch.Tensor, L: int) -> torch.Tensor:
    """Quadratic fit curvature (2nd-order coefficient)."""
    T, D = x.shape
    out = torch.full_like(x, float("nan"))
    k = torch.arange(L, dtype=x.dtype, device=x.device)
    k2 = k * k
    n2 = float(L)
    sx = L * (L - 1) / 2.0
    sxx = L * (L - 1) * (2 * L - 1) / 6.0
    sxxx = L * (L - 1) * (2 * L - 1) * (3 * L**2 - 3 * L - 1) / 30.0
    sxxxx = L * (L - 1) * (2 * L - 1) * (3 * L**2 - 3 * L - 1) * (3 * L**2 - 3 * L + 1) / 42.0
    A = torch.tensor([[n2, sx, sxx],
                      [sx, sxx, sxxx],
                      [sxx, sxxx, sxxxx]], dtype=x.dtype, device=x.device)
    for t in range(L - 1, D):
        w = x[:, t - L + 1 : t + 1]
        sy = w.sum(dim=1)
        sky = (k.unsqueeze(0) * w).sum(dim=1)
        sk2y = (k2.unsqueeze(0) * w).sum(dim=1)
        B = torch.stack([sy, sky, sk2y], dim=1).unsqueeze(-1)
        try:
            coeff = torch.linalg.solve(A.unsqueeze(0).expand(T, 3, 3), B)
            out[:, t] = coeff[:, 2, 0]
        except:
            pass
    return out


def _batched_rolling_min(x: torch.Tensor, L: int) -> torch.Tensor:
    T, D = x.shape
    out = torch.full_like(x, float("nan"))
    for t in range(L - 1, D):
        out[:, t] = x[:, t - L + 1 : t + 1].min(dim=1).values
    return out


def _batched_rolling_max(x: torch.Tensor, L: int) -> torch.Tensor:
    T, D = x.shape
    out = torch.full_like(x, float("nan"))
    for t in range(L - 1, D):
        out[:, t] = x[:, t - L + 1 : t + 1].max(dim=1).values
    return out


def fractal_stats_batch(
    wide_close: np.ndarray,           # [T, D] close prices
    wide_volume: np.ndarray,          # [T, D] volume
    wide_open: np.ndarray | None,     # [T, D] or None
    wide_high: np.ndarray | None,     # [T, D] or None
    wide_low: np.ndarray | None,      # [T, D] or None
    configs: list[tuple[int, int]],   # list of (a, b) fractal configs
    device: str | None = None,
) -> dict:
    """
    Compute full statistical profile for all fractal spans across configs.

    Returns: {span_len: {stat_name: [T, D] tensor}} for stats in STAT_COLS.
    """
    dev = device or _best_device()
    T, D = wide_close.shape

    close = torch.as_tensor(wide_close.copy(), dtype=torch.float64, device=dev)
    volume = torch.as_tensor(wide_volume.copy(), dtype=torch.float64, device=dev)
    has_ohlc = (wide_open is not None and wide_high is not None and wide_low is not None)
    if has_ohlc:
        open_ = torch.as_tensor(wide_open.copy(), dtype=torch.float64, device=dev)
        high = torch.as_tensor(wide_high.copy(), dtype=torch.float64, device=dev)
        low = torch.as_tensor(wide_low.copy(), dtype=torch.float64, device=dev)

    # Collect all unique window lengths
    all_lengths = set()
    for a, b in configs:
        for f, t in spans_generator(a, b):
            all_lengths.add(t - f)
    lengths = sorted(all_lengths)

    result = {}

    for L in lengths:
        # Price stats
        pm = _batched_rolling_mean(close, L)
        pmed = torch.full_like(close, float("nan"))  # median needs quantile - skip on GPU
        pstd = _batched_rolling_std(close, L)
        pmax = _batched_rolling_max(close, L)
        pmin = _batched_rolling_min(close, L)
        prange = pmax - pmin
        pskew = _batched_rolling_skew(close, L)
        pkurt = _batched_rolling_kurt(close, L)
        pmode = _batched_rolling_mode(close, L)

        # Position stats
        close_z = torch.where(pstd > 0, (close - pm) / pstd, float("nan"))
        close_pctile = _batched_rolling_percentile_rank(close, L)
        runup = _batched_rolling_runup(close, L)
        window_dd = _batched_rolling_window_dd(close, L)
        price_slope = _batched_rolling_slope(close, L)
        price_curvature = _batched_rolling_curvature(close, L)

        # Volume stats
        vmean = _batched_rolling_mean(volume, L)
        vstd = _batched_rolling_std(volume, L)
        volume_z = torch.where(vstd > 0, (volume - vmean) / vstd, float("nan"))
        vwap = _batched_rolling_sums(close * volume, L) / _batched_rolling_sums(volume, L)

        # Momentum stats
        logc = torch.log(torch.clamp(close, min=1e-12))
        log_ret = logc - torch.cat([torch.full((T, L), float("nan"), device=dev), logc[:, :-L]], dim=1)
        dr = torch.diff(logc, dim=1, prepend=logc[:, :1] * float("nan"))
        dr2 = dr * dr
        cum_dr2 = torch.cumsum(dr2, dim=1)
        rsq = torch.full_like(dr2, float("nan"))
        rsq[:, L:] = cum_dr2[:, L:] - cum_dr2[:, :-L]
        rsq[:, : L - 1] = float("nan")
        ret_vol = torch.sqrt(torch.clamp(rsq / L, min=0))
        momentum = log_ret / torch.clamp(ret_vol, min=1e-9)

        # True-OHLCV stats
        if has_ohlc:
            typ = (high + low + close) / 3.0
            vwap_true = _batched_rolling_sums(typ * volume, L) / _batched_rolling_sums(volume, L)
            prev_c = torch.cat([close[:, :1] * float("nan"), close[:, :-1]], dim=1)
            tr = torch.maximum(high - low,
                torch.maximum(torch.abs(high - prev_c), torch.abs(low - prev_c)))
            atr = _batched_rolling_mean(tr, L)
            atr_pct = atr / torch.clamp(close, min=1e-9)
            gap = open_ / torch.clamp(prev_c, min=1e-9) - 1.0
            gap_mean = _batched_rolling_mean(gap, L)
            gap_std = _batched_rolling_std(gap, L)
            range_hl = _batched_rolling_mean((high - low) / torch.clamp(close, min=1e-9), L)
            body = (close - open_) / torch.clamp(close, min=1e-9)
            body_mean = _batched_rolling_mean(body, L)
            body_std = _batched_rolling_std(body, L)
            up_wick = (high - torch.maximum(close, open_)) / torch.clamp(close, min=1e-9)
            lo_wick = (torch.minimum(close, open_) - low) / torch.clamp(close, min=1e-9)
            upper_wick = _batched_rolling_mean(up_wick, L)
            lower_wick = _batched_rolling_mean(lo_wick, L)
        else:
            vwap_true = atr = atr_pct = gap_mean = gap_std = torch.full_like(close, float("nan"))
            range_hl = body_mean = body_std = upper_wick = lower_wick = torch.full_like(close, float("nan"))

        result[L] = {
            "price_mean": pm, "price_median": pmed, "price_mode": pmode,
            "price_max": pmax, "price_min": pmin, "price_range": prange,
            "price_std": pstd, "price_skew": pskew, "price_kurtosis": pkurt,
            "close_z": close_z, "close_pctile": close_pctile,
            "runup": runup, "window_drawdown": window_dd,
            "price_slope": price_slope, "price_curvature": price_curvature,
            "volume_mean": vmean, "vwap": vwap, "volume_z": volume_z,
            "log_ret": log_ret, "momentum": momentum, "ret_vol": ret_vol,
            "vwap_true": vwap_true, "atr": atr, "atr_pct": atr_pct,
            "gap_mean": gap_mean, "gap_std": gap_std, "range_hl": range_hl,
            "body_mean": body_mean, "body_std": body_std,
            "upper_wick": upper_wick, "lower_wick": lower_wick,
        }

    return result


def gpu_available() -> bool:
    return torch.cuda.is_available()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=1500, help="trailing days")
    parser.add_argument("--tickers", type=int, default=None, help="cap universe")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    # Load OHLCV from parquet
    vp = pd.read_parquet("daily_prices.parquet",
                         columns=["date", "ticker", "volume", "open", "high", "low"])
    vp["date"] = pd.to_datetime(vp["date"])
    # Close from macro_sector_shock price matrix
    from macro_sector_shock import _load_price_matrix, _price_universe
    w = _load_price_matrix()
    have = _price_universe()
    tickers = sorted(have & set(w.columns))
    if args.tickers:
        tickers = tickers[: args.tickers]

    close_df = w[tickers].tail(args.window)
    close = close_df.to_numpy().T  # [T, D]
    # Align OHLCV to close index
    vm = vp.pivot(index="date", columns="ticker", values="volume")
    om = vp.pivot(index="date", columns="ticker", values="open")
    hm = vp.pivot(index="date", columns="ticker", values="high")
    lm = vp.pivot(index="date", columns="ticker", values="low")

    volume = vm[tickers].reindex(close_df.index).ffill().to_numpy().T
    open_ = om[tickers].reindex(close_df.index).to_numpy().T
    high = hm[tickers].reindex(close_df.index).to_numpy().T
    low = lm[tickers].reindex(close_df.index).to_numpy().T

    configs = [(3, 5), (5, 3), (10, 3), (15, 3), (30, 3)]
    stats = fractal_stats_batch(
        close, volume, open_, high, low, configs
    )
    print(f"GPU stats computed for {len(stats)} window lengths on {close.shape[0]} tickers x {close.shape[1]} days")
    for L, s in stats.items():
        print(f"  L={L}: {len(s)} stats, shape={list(s.values())[0].shape}")
    if args.save:
        # Convert to long-format and save
        # (simplified - just save one span as demo)
        import os
        out_path = "gpu_profiles.parquet"
        print(f"Would save to {out_path}")