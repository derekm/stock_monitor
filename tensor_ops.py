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


def rolling_sum(arr: np.ndarray, window: int, device: str | None = None) -> np.ndarray:
    """Rolling sum over last axis (time). arr: [T, D] or [D]."""
    dev = device or _best_device()
    if is_gpu(dev):
        t = _to_tensor(arr, dev)
        cum = torch.cumsum(t, dim=-1)
        out = torch.full_like(t, float("nan"))
        if t.ndim == 1:
            # out[W-1:] = cum[W-1:] - cum[:-W] (with cum[-1]=0 for first W-1)
            out[window - 1:] = cum[window - 1:] - torch.cat([torch.zeros(1, device=dev), cum[:-window]])
        else:
            # out[:, W-1:] = cum[:, W-1:] - cum[:, :-W] (with zeros on left)
            left = torch.zeros((t.shape[0], 1), device=dev)
            out[:, window - 1:] = cum[:, window - 1:] - torch.cat([left, cum[:, :-window]], dim=1)
        return _to_numpy(out)
    # CPU fallback
    cum = np.nancumsum(arr, axis=-1)
    out = np.full_like(arr, np.nan)
    if arr.ndim == 1:
        out[window - 1:] = cum[window - 1:] - np.concatenate([np.zeros(1), cum[:-window]])
    else:
        out[:, window - 1:] = cum[:, window - 1:] - np.hstack([np.zeros((arr.shape[0], 1)), cum[:, :-window]])
    return out


def rolling_mean(arr: np.ndarray, window: int, device: str | None = None) -> np.ndarray:
    return rolling_sum(arr, window, device) / window


def rolling_std(arr: np.ndarray, window: int, device: str | None = None) -> np.ndarray:
    """Rolling std using Welford / cumsum of squares."""
    dev = device or _best_device()
    if is_gpu(dev):
        t = _to_tensor(arr, dev)
        t2 = t * t
        cum1 = torch.cumsum(t, dim=-1)
        cum2 = torch.cumsum(t2, dim=-1)
        out = torch.full_like(t, float("nan"))
        if t.ndim == 1:
            s1 = cum1[window - 1:] - torch.cat([torch.zeros(1, device=dev), cum1[:-window]])
            s2 = cum2[window - 1:] - torch.cat([torch.zeros(1, device=dev), cum2[:-window]])
        else:
            left = torch.zeros((t.shape[0], 1), device=dev)
            s1 = cum1[:, window - 1:] - torch.cat([left, cum1[:, :-window]], dim=1)
            s2 = cum2[:, window - 1:] - torch.cat([left, cum2[:, :-window]], dim=1)
        mean = s1 / window
        var = torch.clamp(s2 / window - mean * mean, min=0)
        out_val = torch.sqrt(var)
        if t.ndim == 1:
            out[window - 1:] = out_val
        else:
            out[:, window - 1:] = out_val
        return _to_numpy(out)
    # CPU
    cum1 = np.nancumsum(arr, axis=-1)
    cum2 = np.nancumsum(arr * arr, axis=-1)
    out = np.full_like(arr, np.nan)
    if arr.ndim == 1:
        s1 = cum1[window - 1:] - np.concatenate([np.zeros(1), cum1[:-window]])
        s2 = cum2[window - 1:] - np.concatenate([np.zeros(1), cum2[:-window]])
        mean = s1 / window
        var = np.maximum(s2 / window - mean * mean, 0)
        out[window - 1:] = np.sqrt(var)
    else:
        s1 = cum1[:, window - 1:] - np.hstack([np.zeros((arr.shape[0], 1)), cum1[:, :-window]])
        s2 = cum2[:, window - 1:] - np.hstack([np.zeros((arr.shape[0], 1)), cum2[:, :-window]])
        mean = s1 / window
        var = np.maximum(s2 / window - mean * mean, 0)
        out[:, window - 1:] = np.sqrt(var)
    return out


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
        t = _to_tensor(arr, dev)
        # y = rolling window values
        # sum(y) = rolling_sum(arr, n)
        # sum(xy) = rolling_sum(arr * x, n) where x = [0,1,...,n-1]
        idx = torch.arange(t.shape[-1], dtype=torch.float32, device=dev)
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
        b = _to_tensor(bench, dev)
        a = _to_tensor(arr, dev)
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