#!/usr/bin/env python3
"""fractal_windows_gpu.py — GPU-batched fractal momentum (torch, scatter-gather).

The fractal-of-sliding-windows scheme (US20120253946A1 FIGS 28-29) is naturally
scatter-gatherable: each ticker is an independent unit, and all rolling-window
statistics can be computed with batched CUMSUM on a [tickers x days] tensor —
no per-ticker Python loops, no per-day polyfit.

Batched closed-form (all window lengths L in the fractal):
  trailing L-day log return: logp[t] - logp[t-L]
  rolling sum of y over window   = cumsum[y][t] - cumsum[y][t-L]
  rolling sum of k*y over window = cumsum[k*y][t] - cumsum[k*y][t-L]
  slope = [L*Sxy - Sx*Sy] / [L*Sxx - Sx^2]   (Sx, Sxx constant per L)
  vol    = sqrt of rolling variance of daily returns (cumsum-based)

This turns the whole universe into ONE batched tensor op per span length, run on
GPU when available (fallback to CPU). Memory: [T x days] float32 is tiny for the
universe (~583 x ~10k = 23MB).

Usage (as a module):
  from fractal_windows_gpu import fractal_batch
  out = fractal_batch(wide_logp, a=30, b=3)   # -> dict of (span_len -> [T x days] arrays)
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False

from fractal_windows import spans_generator
# Device selection is centralized in tensor_ops — do not reimplement it here.
from tensor_ops import _best_device, is_gpu
from tensor_ops import gpu_available as _to_gpu_available


def _batched_rolling_sums(logp_t: "torch.Tensor", L: int) -> "torch.Tensor":
    """Batched rolling L-day sum of a [T, days] tensor via cumsum, entirely on-device.
    Returns same shape (NaN before L-1)."""
    cum = torch.cumsum(logp_t, dim=1)
    out = cum.clone()
    out[:, L:] = out[:, L:] - cum[:, :-L]
    out[:, : L - 1] = float("nan")
    return out


def fractal_batch(wide_logp: np.ndarray, a: int = 30, b: int = 3,
                  device: str | None = None) -> dict:
    """wide_logp: [T tickers, D days] log-prices (forward-filled, NaN-free).
    Returns {span_len: {ret, slope, momentum, uptrend} as [T, D] tensors}.

    Memory semantics: `wide_logp` is moved to the device once (a copy on CUDA,
    a view on CPU). The fractal span tuples are integers (loop indices, no
    memory). All rolling/intermediate tensors are freshly allocated on-device.
    No host<->device round-trips inside the per-length loop.
    """
    dev = device or _best_device()
    T, D = wide_logp.shape
    spans = spans_generator(a, b)
    lengths = sorted({t - f for f, t in spans})

    logp = torch.as_tensor(wide_logp, dtype=torch.float32, device=dev)
    idx = torch.arange(D, dtype=torch.float32, device=dev)
    k_y = idx * logp  # [T, D]  k * logp
    dr = torch.diff(logp, dim=1, prepend=logp[:, :1] * float("nan"))  # daily ret

    result = {}
    for L in lengths:
        sy = _batched_rolling_sums(logp, L)
        sky = _batched_rolling_sums(k_y, L)
        start = idx - (L - 1)
        sxy = sky - start * sy
        sx = L * (L - 1) / 2.0
        sxx = L * (L - 1) * (2 * L - 1) / 6.0
        denom = L * sxx - sx * sx
        slope = (L * sxy - sx * sy) / denom
        # trailing L-day log return = logp[t] - logp[t-L]
        ret = logp - torch.cat([torch.full((T, L), float("nan"), device=dev), logp[:, :-L]], dim=1)
        # rolling vol of daily returns over L days (cumsum of squares)
        dr2 = dr * dr
        cum_dr = torch.cumsum(dr, dim=1)
        cum_dr2 = torch.cumsum(dr2, dim=1)
        rsum = torch.full_like(dr, float("nan"))
        rsum[:, L:] = cum_dr[:, L:] - cum_dr[:, :-L]
        rsum[:, : L - 1] = float("nan")
        rsq = torch.full_like(dr2, float("nan"))
        rsq[:, L:] = cum_dr2[:, L:] - cum_dr2[:, :-L]
        rsq[:, : L - 1] = float("nan")
        var = (rsq - rsum * rsum / L) / (L - 1)
        vol = torch.sqrt(torch.clamp(var, min=0))
        momentum = ret / torch.clamp(vol, min=1e-9)
        uptrend = (ret > 0) & (slope > 0)

        # attach every span of this length
        for (f, t) in spans:
            if t - f != L:
                continue
            result[(f, t)] = {
                "ret": ret, "slope": slope, "momentum": momentum,
                "uptrend": uptrend, "vol": vol,
            }
    return result


def gpu_available() -> bool:
    """True when any accelerator is usable (CUDA or DirectML).

    Delegates to tensor_ops. The old implementation checked
    `torch.cuda.is_available()` only, so a DirectML-only host (Intel Xe)
    reported False even though the GPU path worked.
    """
    return _to_gpu_available()


def fractal_consensus_batch(res: dict, T: int, D: int, device: str | None = None
                            ) -> dict:
    """On-device consensus over all fractal spans: mean of each stat across the
    span axis. No groupby — the span dimension is a fixed, known axis, so this
    is a torch stack + mean over dim 0. Returns dict of [T, D] tensors:
    frac_uptrend, mean_momentum, mean_ret, mean_slope, n_spans."""
    dev = device or _best_device()
    spans = list(res.keys())
    if not spans:
        return {}
    # stack each stat across spans -> [n_spans, T, D]
    frac = torch.stack([res[s]["uptrend"].float() for s in spans], 0)
    mom = torch.stack([res[s]["momentum"].float() for s in spans], 0)
    ret = torch.stack([res[s]["ret"].float() for s in spans], 0)
    slp = torch.stack([res[s]["slope"].float() for s in spans], 0)
    # NaN-safe mean (ignore NaN entries)
    def nanmean(x):
        mask = torch.isfinite(x)
        s = torch.where(mask, x, torch.zeros_like(x)).sum(0)
        n = mask.sum(0).clamp(min=1)
        return s / n
    return {
        "frac_uptrend": nanmean(frac),
        "mean_momentum": nanmean(mom),
        "mean_ret": nanmean(ret),
        "mean_slope": nanmean(slp),
        "n_spans": torch.full((T, D), float(len(spans)), device=dev),
    }
