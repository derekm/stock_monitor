# statistical_profiler.py — a TRUE statistical profiler over fractal windows

## Why it exists (rationale)

The fractal-of-sliding-windows scheme (US20120253946A1) tiles a price range into
every aligned granularity, but the original `fractal_signal_vec` computed only
**momentum** stats (log-return, slope, ret/vol). That throws away most of what a
window's price path reveals. This module widens each fractal window into a full
**statistical profile** and **persists it**, so downstream experiments run on the
saved results instead of re-running price history every time.

Two design goals:
1. **True profiler** — capture a wide array of per-window statistics (mean,
   median, mode, VWAP, skew, kurtosis, percentile rank, z-score, run-up,
   drawdown, slope, curvature, volume stats), not just momentum.
2. **Preserve results** — write a long-format `fractal_profiles.parquet` keyed by
   (ticker, date, span) once; `backtest_price_vs_momentum.py` reads it and never
   recomputes window statistics from raw prices.

## What each window reports

For every trailing fractal window (span_len trading days ending at `date`):

| family | columns |
|--------|---------|
| price level | `price_mean, price_median, price_mode, price_max, price_min, price_range, price_std` |
| price shape | `price_skew, price_kurtosis, price_slope, price_curvature` |
| current position | `close_z, close_pctile, runup, window_drawdown` |
| volume | `volume_mean, vwap, volume_z` |
| momentum (kept) | `log_ret, momentum (ret/vol), ret_vol` |

Notes:
- `vwap` = Σ(close·vol)/Σvol — a close-based approximation, since true
  OHLC (high/low) is ~absent from the stored history (~0.8% coverage).
- `price_mode` is a histogram-based approximation (normalized, 9 bins).
- All stats are **point-in-time**: only data up to the window end is used.

## Usage

```bash
python statistical_profiler.py --window 1500 --save   # full universe
python statistical_profiler.py --tickers 50 --save    # subset
```

## Outputs

- `fractal_profiles.parquet` — long format: `ticker, date, span_from, span_to,
  span_len, close,` + all stat columns above.

## Related

- `backtest_price_vs_momentum.py` — reads the profiles; does PRICE-based signals
  predict forward returns better than MOMENTUM-based ones?
- `fractal_windows.py` — `spans_generator` (the span geometry this profiles)
- `fractal_windows_backtest_gpu.py` — the momentum-only GPU universe backtest
