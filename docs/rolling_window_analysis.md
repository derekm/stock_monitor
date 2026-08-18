# rolling_window_analysis.py

Rolling vol, beta, Sharpe, max-drawdown, and dual-screen stability — now with GPU-accelerated cumsum and resumable checkpoints.

## Why it exists (rationale)

Single-point risk stats hide regime dependence. This rolls vol/beta/Sharpe/DD (and dual-screen pass stability) over a trailing window per ticker, so you can see whether a name's risk profile is stable or deteriorating — input to sizing and the buy decision.

## Usage

```bash
python rolling_window_analysis.py --universe all --window 63 --save
python rolling_window_analysis.py --universe portfolio --window 252 --save
```

Flags:
- `--universe`: `all`, `portfolio`, `index_member`, or comma-separated tickers (default: `all`)
- `--window`: Rolling window in days (default: 63)
- `--save`: Write output parquet
- `--gpu`: Force GPU device (`cuda`, `directml`, `cpu`)

Reads `daily_prices.parquet`, `monitored_stocks.parquet`, `portfolio_holdings.parquet`, `preferred_metrics.parquet`.

## Key Features (2026-08)

- **Device-resident kernels**: the `rolling_*_t` functions take and return torch tensors and never sync to host, so a multi-op pipeline uploads once and downloads once. The numpy-facing `rolling_*` API is unchanged and still returns numpy. This matters more than the kernels themselves: on a 300x1500 float64 panel (3.6 MB, MX550), 25 numpy-in/numpy-out calls took **687 ms of which `.cpu()` alone was 78%**, while the same 25 ops kept resident took **23 ms — ~30x**, entirely transfer/sync overhead rather than compute. A parallel `_gpu` module is the wrong fix for this; a tensor-in/tensor-out core is the right one, which is the one thing the deleted `statistical_profiler_gpu.py` had structurally right even though its numerics were wrong.
- **VRAM is bounded by the window buffer, not the panel**: a `[T, D-L+1, L]` float64 reduction temporary is 0.75 GB at 800x2000xL60 and several are live at once, which overruns a 2.15 GB MX550 and made the GPU **slower than CPU (25.98s vs 12.72s)**. `statistical_profiler` chunks over tickers to fit; after chunking the same case is 2.76s. The MX550 is a *discrete* GPU (`is_integrated=0`), so there is no unified host/device address space to exploit — the win is residency, not zero-copy.
- **DirectML cannot host the float64 resident path**: torch-directml implements `sqrt`, `pow`, `clamp`, `sort`, `isnan` and `nanquantile` for **float32 only**; in float64 each raises `"The parameter is incorrect."` `t**0.5`, `exp(0.5*log(t))` and sort-based selection all fail identically, so there is no workaround. Float32 is not an acceptable fallback — a rolling std on price-level data (~7.1e6) is off by **0.20 absolute**. `tensor_ops.resident_device()` therefore routes DirectML to CPU, and `_isnan_t` uses `t != t` (the IEEE test, which does work on DirectML f64). Measured: 800x2000 CPU 13.93s / CUDA 2.76s / DirectML 14.49s (ran on CPU).
- **Centralized device handling**: `tensor_ops` is the single source of truth for GPU/CPU selection (`get_device`/`best_device`, `resolve_device`, `resident_device`, `is_gpu`, `supports_f64_rolling`, `gpu_available`, `device_name`, `to_device`). `fractal_windows`, `fractal_windows_backtest_gpu` and `backtest_coiled_spring` import from it (the separate `fractal_windows_gpu.py` / `statistical_profiler_gpu.py` modules were deleted 2026-08 — a parallel `_gpu` file only duplicated the device ladder and drifted from it) rather than reimplementing the CUDA→DirectML→CPU ladder. `_best_device()` returns a `torch.device` **object**, never a string — `torch.device("cpu") != "cpu"` is truthy, and string guards silently routed CPU work down the GPU branch.
- **No local CPU fallbacks**: every rolling op in `tensor_ops` already has an internal numpy path, so callers must not wrap it in their own try/except GPU→CPU ladder.

- **GPU acceleration**: Uses `tensor_ops.rolling_cumsum_2d` for vectorized cumsum on CUDA or DirectML (Intel Xe). Falls back to CPU NumPy automatically.
- **Resumable checkpoints**: Uses `resumable_job.JobCheckpoint` — skips tickers already computed (prints `=== Rolling 63d SKIP N tickers already complete ===`).
- **Universe from daily_prices**: No longer uses `monitored_stocks.parquet` as universe source.
- **Output**: `rolling_window_metrics.parquet` with columns: `ticker`, `as_of_date`, `window`, `vol`, `beta`, `sharpe`, `max_dd`, `dual_screen_stable`, `data_provenance`.

## Performance

| Universe | Tickers | Window | CUDA (MX550) | DirectML (Xe) | CPU |
|----------|---------|--------|--------------|---------------|-----|
| all | ~10k | 63 | ~0.12s | ~0.20s | ~0.03s |
| all | ~10k | 252 | ~0.16s | ~0.33s | ~0.44s |

## Outputs

- `rolling_window_metrics.parquet` — per-ticker rolling risk metrics (Schema family: `summary_metrics` — see [SCHEMAS.md](SCHEMAS.md).)
- Checkpoint: `backfill_checkpoints/rolling_window_analysis_daily_prices.json`

## Related programs

- [risk_enrich.md](risk_enrich.md) / [risk_metrics_ext.md](risk_metrics_ext.md)
- [preferred_metrics.md](preferred_metrics.md)
- [momentum_analytics.md](momentum_analytics.md)
- [tensor_ops.py](tensor_ops.py) — unified GPU/CPU rolling operations
- [resumable_job.md](resumable_job.md) — checkpoint framework