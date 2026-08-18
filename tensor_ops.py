#!/usr/bin/env python3
"""tensor_ops.py — GPU/CPU unified rolling linear algebra.

Provides batched rolling operations for [T, D] tensors (T=tickers, D=days):
- rolling_sum, rolling_mean, rolling_std
- rolling_slope (linear regression slope per window)
- rolling_beta (cov/var against a benchmark series)

Falls back to CPU numpy if torch unavailable or device=cpu.
"""

from __future__ import annotations
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _best_device():
    """CUDA, then DirectML, then CPU. Always returns a torch.device.

    Canonical device selection for the whole repo — do NOT reimplement this in
    individual scripts. Import `get_device` (or `best_device`) from here.
    Returning a device OBJECT (not a string) matters: `torch.device("cpu")`
    compares unequal to the string "cpu", so string-based guards silently send
    CPU work down the GPU branch.
    """
    if not _HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml  # type: ignore
        return torch_directml.device()
    except Exception:  # noqa: BLE001 - any DirectML import/init failure -> CPU
        pass
    return torch.device("cpu")


def is_gpu(device=None) -> bool:
    """True when `device` (or the auto-selected one) is a real accelerator."""
    dev = device if device is not None else _best_device()
    if not _HAS_TORCH or not isinstance(dev, torch.device):
        return False
    return dev.type != "cpu"


def gpu_available() -> bool:
    """True when any accelerator (CUDA or DirectML) is usable."""
    return is_gpu()


def device_name(device=None) -> str:
    """Human-readable device label for logs."""
    dev = device if device is not None else _best_device()
    if not _HAS_TORCH or not isinstance(dev, torch.device):
        return "cpu"
    if dev.type == "cuda":
        try:
            return f"cuda:{torch.cuda.get_device_name(0)}"
        except Exception:  # noqa: BLE001
            return "cuda"
    return str(dev)


def resolve_device(device=None):
    """Normalize None / "auto" / "cpu" / "cuda" / device -> a torch.device.

    Use this in CLI entry points so the device is resolved ONCE and logs
    report what will actually be used.
    """
    if device is None or (isinstance(device, str) and device == "auto"):
        return _best_device()
    if _HAS_TORCH and isinstance(device, str):
        return torch.device(device)
    return device


def to_device(arr, device=None, dtype=None):
    """Move a numpy array onto `device` as a tensor, or return it unchanged on CPU.

    Central conversion helper so scripts stop hand-rolling
    `torch.as_tensor(..., device=...)` behind their own try/except.
    """
    dev = resolve_device(device)
    if not is_gpu(dev):
        return arr
    return torch.as_tensor(arr, dtype=dtype or torch.float32, device=dev)


def _to_tensor(arr: np.ndarray, device, dtype=torch.float32):
    if _HAS_TORCH and isinstance(device, torch.device) and device.type != "cpu":
        return torch.as_tensor(arr, dtype=dtype, device=device)
    if _HAS_TORCH and isinstance(device, str) and device != "cpu":
        return torch.as_tensor(arr, dtype=dtype, device=device)
    return arr  # numpy array


def _to_numpy(t):
    if _HAS_TORCH and isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    return t


def _nan_to_zero_mask(t):
    """Return (values-with-NaN-as-0, valid-count-mask) for NaN-safe cumsums.

    torch.cumsum PROPAGATES NaN: one NaN poisons every later element. numpy's
    nancumsum treats NaN as 0. Without this the GPU and CPU paths diverge
    completely on any series with leading NaN (e.g. a rolling output fed into
    another rolling op) -- measured 22,500 NaN on GPU vs 100 on CPU.
    """
    nan = torch.isnan(t)
    return torch.where(nan, torch.zeros_like(t), t), (~nan).to(t.dtype)


def _valid_count(arr: np.ndarray, window: int, device=None) -> np.ndarray:
    """Count of non-NaN observations in each trailing window (same shape as arr).

    Used to honour `min_periods`: a window containing fewer than the required
    number of real observations must yield NaN, not a sum over zero-filled
    NaNs. Without this, rolling_sum/mean/std silently treat NaN as 0 and return
    a value where pandas/polars return NaN -- which is how a 72%-NaN price
    panel produced bb_width errors of 3.45 absolute.
    """
    mask = (~np.isnan(np.asarray(arr, dtype=float))).astype(float)
    return _rolling_sum_raw(mask, window, device=device)


def _rolling_sum_raw(arr: np.ndarray, window: int, device=None) -> np.ndarray:
    """Rolling sum treating NaN as 0, with NO min_periods masking."""
    dev = device if device is not None else _best_device()
    if is_gpu(dev):
        t = _to_tensor(arr, dev, dtype=torch.float64)
        t, _ = _nan_to_zero_mask(t)
        cum = torch.cumsum(t, dim=-1)
        out = torch.full_like(t, float("nan"))
        if t.ndim == 1:
            out[window - 1:] = cum[window - 1:] - torch.cat([torch.zeros(1, device=dev, dtype=t.dtype), cum[:-window]])
        else:
            left = torch.zeros((t.shape[0], 1), device=dev, dtype=t.dtype)
            out[:, window - 1:] = cum[:, window - 1:] - torch.cat([left, cum[:, :-window]], dim=1)
        return _to_numpy(out)
    cum = np.nancumsum(arr, axis=-1)
    out = np.full_like(np.asarray(arr, dtype=float), np.nan)
    if np.ndim(arr) == 1:
        out[window - 1:] = cum[window - 1:] - np.concatenate([np.zeros(1), cum[:-window]])
    else:
        out[:, window - 1:] = cum[:, window - 1:] - np.hstack([np.zeros((np.shape(arr)[0], 1)), cum[:, :-window]])
    return out


def rolling_sum(arr: np.ndarray, window: int, device: str | None = None,
                min_periods: int | None = None) -> np.ndarray:
    """Rolling sum over last axis (time). arr: [T, D] or [D].

    NaN handling matches pandas/polars: NaNs contribute 0, but a window with
    fewer than `min_periods` real observations yields NaN (default: the full
    window must be observed, i.e. min_periods=window).
    """
    dev = device or _best_device()
    mp = window if min_periods is None else min_periods
    out = _rolling_sum_raw(arr, window, device=dev)
    if mp > 1:
        cnt = _valid_count(arr, window, device=dev)
        out = np.where(cnt >= mp, out, np.nan)
    return out


def rolling_mean(arr: np.ndarray, window: int, device: str | None = None,
                 min_periods: int | None = None) -> np.ndarray:
    """Rolling mean over the last axis, NaN-aware like pandas/polars.

    Divides by the number of OBSERVED values in the window, not the window
    width, and returns NaN when fewer than `min_periods` are observed.

    Uses a strided window sum rather than a cumsum difference: on a real price
    panel (values to 7e6, cumsum to 2e9) the cumsum form lost ~1.7e-07 between
    devices, which is enough to flip knife-edge signal comparisons.
    """
    dev = device or _best_device()
    a = np.asarray(arr, dtype=float)
    mp = window if min_periods is None else min_periods
    cnt = _valid_count(a, window, device=dev)

    a2 = a if a.ndim == 2 else a[None, :]
    T, D = a2.shape
    res = np.full((T, D), np.nan, dtype=float)
    if D >= window:
        if is_gpu(dev):
            t = _to_tensor(np.ascontiguousarray(a2), dev, dtype=torch.float64)
            wv = t.unfold(1, window, 1)
            valid = ~torch.isnan(wv)
            n = valid.sum(-1).to(t.dtype)
            s = torch.where(valid, wv, torch.zeros_like(wv)).sum(-1)
            res[:, window - 1:] = _to_numpy(s / torch.clamp(n, min=1.0))
        else:
            from numpy.lib.stride_tricks import sliding_window_view
            wv = sliding_window_view(a2, window, axis=1)
            valid = ~np.isnan(wv)
            n = valid.sum(-1).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                res[:, window - 1:] = (np.where(valid, wv, 0.0).sum(-1)
                                       / np.maximum(n, 1.0))
    out = res if a.ndim == 2 else res[0]
    return np.where(cnt >= mp, out, np.nan)


def rolling_std(arr: np.ndarray, window: int, device: str | None = None,
                ddof: int = 0, min_periods: int | None = None) -> np.ndarray:
    """Rolling std over the last axis, NaN-aware like pandas/polars.

    ddof=0 is the population form (torch's `unbiased=False`); ddof=1 is the
    sample form used by polars `rolling_std` and pandas `.rolling().std()`.
    Getting this wrong silently rescales Bollinger-style bands by
    sqrt(w/(w-1)) -- for w=20 that is 2.6%, enough to move signal thresholds.

    Both device paths share one formulation, so GPU and CPU cannot drift apart.

    NUMERICAL NOTE: the textbook cumsum-of-squares identity
    (sum(x^2) - n*mean^2) is catastrophically unstable on real price panels --
    prices up to 7e6 push sum(x^2) to ~6e15, which exhausts float64's ~16
    significant digits and left a measured 0.68 absolute error between devices.
    This uses a windowed two-pass form instead: subtract a per-window mean
    before squaring, so the magnitudes stay O(deviation) rather than O(level).
    """
    dev = device or _best_device()
    a = np.asarray(arr, dtype=float)
    mp = window if min_periods is None else min_periods
    cnt = _valid_count(a, window, device=dev)

    # Stable two-pass: centre each window on its own mean via a strided view.
    a2 = a if a.ndim == 2 else a[None, :]
    T, D = a2.shape
    res = np.full((T, D), np.nan, dtype=float)
    if D >= window:
        if is_gpu(dev):
            t = _to_tensor(np.ascontiguousarray(a2), dev, dtype=torch.float64)
            wv = t.unfold(1, window, 1)                       # [T, W, window]
            valid = ~torch.isnan(wv)
            n = valid.sum(-1).to(t.dtype)
            filled = torch.where(valid, wv, torch.zeros_like(wv))
            mean = filled.sum(-1) / torch.clamp(n, min=1.0)
            dev_sq = torch.where(valid, (wv - mean.unsqueeze(-1)) ** 2,
                                 torch.zeros_like(wv)).sum(-1)
            denom = torch.clamp(n - ddof, min=1e-12)
            val = torch.sqrt(torch.clamp(dev_sq, min=0.0) / denom)
            res[:, window - 1:] = _to_numpy(val)
        else:
            from numpy.lib.stride_tricks import sliding_window_view
            wv = sliding_window_view(a2, window, axis=1)
            valid = ~np.isnan(wv)
            n = valid.sum(-1).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(valid, wv, 0.0).sum(-1) / np.maximum(n, 1.0)
                dev_sq = np.where(valid, (wv - mean[..., None]) ** 2, 0.0).sum(-1)
                res[:, window - 1:] = np.sqrt(np.maximum(dev_sq, 0.0) /
                                              np.maximum(n - ddof, 1e-12))
    out = res if a.ndim == 2 else res[0]
    # need at least ddof+1 real observations for a defined sample std
    return np.where((cnt >= mp) & (cnt > ddof), out, np.nan)


def rolling_reduce(arr: np.ndarray, window: int, op: str = "max",
                   device: str | None = None, min_periods: int | None = None) -> np.ndarray:
    """Rolling max/min/sum over the last axis via a strided unfold.

    arr: [T, D] or [D]. NaN-aware (NaNs are ignored, matching pandas). Uses one
    batched unfold on GPU instead of a per-ticker Python loop.
    """
    dev = device or _best_device()
    mp = window if min_periods is None else min_periods
    a2 = arr if arr.ndim == 2 else arr[None, :]
    T, D = a2.shape
    if D < window:
        out = np.full_like(a2, np.nan, dtype=float)
        return out if arr.ndim == 2 else out[0]

    if is_gpu(dev):
        # float64 keeps max/min bit-exact against the CPU/pandas reference;
        # downstream comparisons here are equality-sensitive.
        t = _to_tensor(np.ascontiguousarray(a2), dev, dtype=torch.float64)
        w = t.unfold(1, window, 1)                       # [T, D-window+1, window]
        nan = torch.isnan(w)
        cnt = (~nan).sum(-1)
        if op == "max":
            filled = torch.where(nan, torch.full_like(w, float("-inf")), w)
            val = filled.max(-1).values
        elif op == "min":
            filled = torch.where(nan, torch.full_like(w, float("inf")), w)
            val = filled.min(-1).values
        else:
            filled = torch.where(nan, torch.zeros_like(w), w)
            val = filled.sum(-1)
        val = torch.where(cnt >= mp, val, torch.full_like(val, float("nan")))
        out = torch.full((T, D), float("nan"), device=val.device, dtype=val.dtype)
        out[:, window - 1:] = val
        res = _to_numpy(out)
    else:
        res = np.full((T, D), np.nan, dtype=float)
        # stride tricks give the same batched view without a Python loop
        from numpy.lib.stride_tricks import sliding_window_view
        w = sliding_window_view(a2.astype(float), window, axis=1)
        cnt = np.sum(~np.isnan(w), axis=-1)
        with np.errstate(all="ignore"):
            if op == "max":
                val = np.nanmax(np.where(np.isnan(w), -np.inf, w), axis=-1)
            elif op == "min":
                val = np.nanmin(np.where(np.isnan(w), np.inf, w), axis=-1)
            else:
                val = np.nansum(w, axis=-1)
        val = np.where(cnt >= mp, val, np.nan)
        res[:, window - 1:] = val
    return res if arr.ndim == 2 else res[0]


def rolling_rank_pct(arr: np.ndarray, window: int, device: str | None = None,
                     min_periods: int | None = None) -> np.ndarray:
    """Rolling percentile rank of the CURRENT value within its trailing window.

    Matches pandas `.rolling(w).rank(pct=True)`: average ranks for ties, then
    divided by the non-NaN count. Batched across the ticker axis.
    """
    dev = device or _best_device()
    mp = window if min_periods is None else min_periods
    a2 = arr if arr.ndim == 2 else arr[None, :]
    T, D = a2.shape
    if D < window:
        out = np.full_like(a2, np.nan, dtype=float)
        return out if arr.ndim == 2 else out[0]

    if is_gpu(dev):
        # float64: tie detection uses exact `==`, so a float32 round-trip can
        # merge/split ties and shift the percentile (measured 2.0e-03 error).
        t = _to_tensor(np.ascontiguousarray(a2), dev, dtype=torch.float64)
        w = t.unfold(1, window, 1)                        # [T, W, window]
        cur = w[..., -1:]                                 # current value
        valid = ~torch.isnan(w)
        cnt = valid.sum(-1)
        less = ((w < cur) & valid).sum(-1)
        equal = ((w == cur) & valid).sum(-1)
        # average rank for ties, as pandas does
        rank = less.to(cur.dtype) + (equal.to(cur.dtype) + 1.0) / 2.0
        pct = rank / cnt.clamp(min=1).to(cur.dtype)
        ok = (cnt >= mp) & ~torch.isnan(cur.squeeze(-1))
        pct = torch.where(ok, pct, torch.full_like(pct, float("nan")))
        out = torch.full((T, D), float("nan"), device=pct.device, dtype=pct.dtype)
        out[:, window - 1:] = pct
        res = _to_numpy(out)
    else:
        from numpy.lib.stride_tricks import sliding_window_view
        res = np.full((T, D), np.nan, dtype=float)
        w = sliding_window_view(a2.astype(float), window, axis=1)
        cur = w[..., -1:]
        valid = ~np.isnan(w)
        cnt = valid.sum(-1)
        less = ((w < cur) & valid).sum(-1)
        equal = ((w == cur) & valid).sum(-1)
        rank = less + (equal + 1.0) / 2.0
        with np.errstate(all="ignore"):
            pct = rank / np.maximum(cnt, 1)
        pct = np.where((cnt >= mp) & ~np.isnan(cur[..., 0]), pct, np.nan)
        res[:, window - 1:] = pct
    return res if arr.ndim == 2 else res[0]


def rolling_moment(arr: np.ndarray, window: int, order: int,
                   device: str | None = None,
                   min_periods: int | None = None) -> np.ndarray:
    """Standardized rolling 3rd (skew) or 4th (kurtosis) moment.

    order=3 -> E[(x-mu)^3]/sigma^3, order=4 -> E[(x-mu)^4]/sigma^4 (raw, NOT
    excess kurtosis). Uses the same windowed two-pass form as rolling_std, so it
    is stable on price-level input where the cumsum-of-powers identity is not.
    """
    if order not in (3, 4):
        raise ValueError("order must be 3 or 4")
    dev = device or _best_device()
    a = np.asarray(arr, dtype=float)
    mp = window if min_periods is None else min_periods
    cnt = _valid_count(a, window, device=dev)

    a2 = a if a.ndim == 2 else a[None, :]
    T, D = a2.shape
    res = np.full((T, D), np.nan, dtype=float)
    if D >= window:
        if is_gpu(dev):
            t = _to_tensor(np.ascontiguousarray(a2), dev, dtype=torch.float64)
            wv = t.unfold(1, window, 1)
            valid = ~torch.isnan(wv)
            n = valid.sum(-1).to(t.dtype).clamp(min=1.0)
            zero = torch.zeros_like(wv)
            mean = torch.where(valid, wv, zero).sum(-1) / n
            d = torch.where(valid, wv - mean.unsqueeze(-1), zero)
            m2 = (d ** 2).sum(-1) / n
            mk = (d ** order).sum(-1) / n
            sig = torch.sqrt(torch.clamp(m2, min=0.0))
            val = mk / torch.clamp(sig ** order, min=1e-300)
            res[:, window - 1:] = _to_numpy(val)
        else:
            from numpy.lib.stride_tricks import sliding_window_view
            wv = sliding_window_view(a2, window, axis=1)
            valid = ~np.isnan(wv)
            n = np.maximum(valid.sum(-1).astype(float), 1.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(valid, wv, 0.0).sum(-1) / n
                d = np.where(valid, wv - mean[..., None], 0.0)
                m2 = (d ** 2).sum(-1) / n
                mk = (d ** order).sum(-1) / n
                sig = np.sqrt(np.maximum(m2, 0.0))
                res[:, window - 1:] = mk / np.maximum(sig ** order, 1e-300)
    out = res if a.ndim == 2 else res[0]
    return np.where(cnt >= mp, out, np.nan)


def rolling_skew(arr, window, device=None, min_periods=None):
    """Rolling skewness (standardized 3rd moment)."""
    return rolling_moment(arr, window, 3, device=device, min_periods=min_periods)


def rolling_kurt(arr, window, device=None, min_periods=None):
    """Rolling kurtosis (standardized 4th moment, NOT excess)."""
    return rolling_moment(arr, window, 4, device=device, min_periods=min_periods)


def rolling_slope(arr: np.ndarray, window: int, device: str | None = None) -> np.ndarray:
    """
    Rolling OLS slope of arr against time index [0, 1, ..., D-1].
    arr: [T, D] or [D]. Returns [T, D] or [D] with NaN before window-1.
    Formula: slope = (n*sum(xy) - sum(x)sum(y)) / (n*sum(x^2) - sum(x)^2)
    where x = 0..window-1 (constant per window), y = rolling window of arr.
    """
    dev = device or _best_device()
    n = window
    # x = 0..n-1, constant
    sx = n * (n - 1) / 2.0
    sxx = n * (n - 1) * (2 * n - 1) / 6.0
    denom = n * sxx - sx * sx
    if denom == 0:
        return np.full_like(arr, np.nan)

    if is_gpu(dev):
        # float64 for the same cumsum-cancellation reason as rolling_sum.
        t = _to_tensor(arr, dev, dtype=torch.float64)
        t, _ = _nan_to_zero_mask(t)  # NaN-consistent with the numpy path
        # y = rolling window values
        # sum(y) = rolling_sum(arr, n)
        # sum(xy) = rolling_sum(arr * x, n) where x = [0,1,...,n-1]
        idx = torch.arange(t.shape[-1], dtype=t.dtype, device=dev)
        xy = t * idx
        cum_y = torch.cumsum(t, dim=-1)
        cum_xy = torch.cumsum(xy, dim=-1)
        out = torch.full_like(t, float("nan"))
        if t.ndim == 1:
            sy = cum_y[n - 1:] - torch.cat([torch.zeros(1, device=dev), cum_y[:-n]])
            sxy = cum_xy[n - 1:] - torch.cat([torch.zeros(1, device=dev), cum_xy[:-n]])
            out[n - 1:] = (n * sxy - sx * sy) / denom
        else:
            left = torch.zeros((t.shape[0], 1), device=dev)
            sy = cum_y[:, n - 1:] - torch.cat([left, cum_y[:, :-n]], dim=1)
            sxy = cum_xy[:, n - 1:] - torch.cat([left, cum_xy[:, :-n]], dim=1)
            out[:, n - 1:] = (n * sxy - sx * sy) / denom
        return _to_numpy(out)
    # CPU
    idx = np.arange(arr.shape[-1], dtype=float)
    xy = arr * idx
    cum_y = np.nancumsum(arr, axis=-1)
    cum_xy = np.nancumsum(xy, axis=-1)
    out = np.full_like(arr, np.nan)
    if arr.ndim == 1:
        sy = cum_y[n - 1:] - np.concatenate([np.zeros(1), cum_y[:-n]])
        sxy = cum_xy[n - 1:] - np.concatenate([np.zeros(1), cum_xy[:-n]])
        out[n - 1:] = (n * sxy - sx * sy) / denom
    else:
        sy = cum_y[:, n - 1:] - np.hstack([np.zeros((arr.shape[0], 1)), cum_y[:, :-n]])
        sxy = cum_xy[:, n - 1:] - np.hstack([np.zeros((arr.shape[0], 1)), cum_xy[:, :-n]])
        out[:, n - 1:] = (n * sxy - sx * sy) / denom
    return out


def rolling_beta(arr: np.ndarray, bench: np.ndarray, window: int, device: str | None = None) -> np.ndarray:
    """
    Rolling beta of each series in arr [T, D] against benchmark bench [D].
    beta = cov(arr, bench) / var(bench) over rolling window.
    """
    dev = device or _best_device()
    n = window
    # Precompute benchmark stats
    if is_gpu(dev):
        # float64 for the same cumsum-cancellation reason as rolling_sum.
        b = _to_tensor(bench, dev, dtype=torch.float64)
        a = _to_tensor(arr, dev, dtype=torch.float64)
        b, _ = _nan_to_zero_mask(b)  # NaN-consistent with the numpy path
        a, _ = _nan_to_zero_mask(a)
        cum_b = torch.cumsum(b, dim=-1)
        cum_b2 = torch.cumsum(b * b, dim=-1)
        cum_a = torch.cumsum(a, dim=-1)
        cum_a2 = torch.cumsum(a * a, dim=-1)
        cum_ab = torch.cumsum(a * b, dim=-1)
        out = torch.full_like(a, float("nan"))
        left = torch.zeros((a.shape[0], 1), device=dev)
        sb = cum_b[n - 1:] - torch.cat([torch.zeros(1, device=dev), cum_b[:-n]])
        sb2 = cum_b2[n - 1:] - torch.cat([torch.zeros(1, device=dev), cum_b2[:-n]])
        sa = cum_a[:, n - 1:] - torch.cat([left, cum_a[:, :-n]], dim=1)
        sa2 = cum_a2[:, n - 1:] - torch.cat([left, cum_a2[:, :-n]], dim=1)
        sab = cum_ab[:, n - 1:] - torch.cat([left, cum_ab[:, :-n]], dim=1)
        mean_b = sb / n
        mean_a = sa / n
        var_b = torch.clamp(sb2 / n - mean_b * mean_b, min=1e-12)
        cov = sab / n - mean_a * mean_b
        out[:, n - 1:] = cov / var_b
        return _to_numpy(out)
    # CPU
    cum_b = np.nancumsum(bench)
    cum_b2 = np.nancumsum(bench * bench)
    cum_a = np.nancumsum(arr, axis=-1)
    cum_a2 = np.nancumsum(arr * arr, axis=-1)
    cum_ab = np.nancumsum(arr * bench, axis=-1)
    out = np.full_like(arr, np.nan)
    sb = cum_b[n - 1:] - np.concatenate([np.zeros(1), cum_b[:-n]])
    sb2 = cum_b2[n - 1:] - np.concatenate([np.zeros(1), cum_b2[:-n]])
    sa = cum_a[:, n - 1:] - np.hstack([np.zeros((arr.shape[0], 1)), cum_a[:, :-n]])
    sa2 = cum_a2[:, n - 1:] - np.hstack([np.zeros((arr.shape[0], 1)), cum_a2[:, :-n]])
    sab = cum_ab[:, n - 1:] - np.hstack([np.zeros((arr.shape[0], 1)), cum_ab[:, :-n]])
    mean_b = sb / n
    mean_a = sa / n
    var_b = np.maximum(sb2 / n - mean_b * mean_b, 1e-12)
    cov = sab / n - mean_a * mean_b
    out[:, n - 1:] = cov / var_b
    return out


# Convenience: auto-detect device
def get_device() -> str:
    return _best_device()


# Public alias — prefer this name in new code.
best_device = get_device