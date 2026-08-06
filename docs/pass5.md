# pass5.py — honest OOS Granite-TTM evaluation

`pass5.py` measures whether Granite-TTM forecasts beat a persistence baseline
*out of sample*. It exists because pass4 measured in-sample error (windows
built and scored on the same data, warm-started from a full-history checkpoint
= memorization, not forecasting).

## The two honest protocols (both temporally disjoint)

- **`trainlast` (default, production regime)**: train on the LAST 10y
  (`RECENT=2520`), test by forecasting the ~10y IMMEDIATELY BEFORE that
  window. Train and test regions are disjoint and separated by a gap — the
  model never sees test prices.
- **`half`**: train on the first half of history, test on the second
  (expanding-origin style). `--cutoff-frac` controls the split point.

## Honesty rules baked in

1. Trained from the IBM base model only (`pretrained=False`) — no
   full-history checkpoint contamination of the holdout.
2. Persistence baseline computed on the SAME test windows (apples-to-apples):
   predict the last context close forward (flatline) and compare MAPE /
   direction on the identical windows.
3. Test windows lie entirely within the held-out region (no straddle leakage).

## Results so far (pass-5 rule: every claimed stat is OOS)

- Direction: model beats persistence on **direction accuracy** (mean ~59.8%
  vs ~34% persistence across the 12/12 series in the earlier sweep).
- Level: model does NOT beat persistence on MAPE (0/12) — it is a direction
  forecaster, not a level forecaster.
- Regime-conditioned results: see `regime_forecast.py` / `regime_forecast_stats.csv`
  (direction edge is regime-dependent; persistence itself is highest in
  high_vol_stress, so per-regime baselines are required).

## Usage

```bash
python pass5.py                              # trainlast, AEP/NVR/FICO, steps=6000
python pass5.py --mode half --cutoff-frac 0.5
python pass5.py --tickers AEP KO XOM --steps 9000 --strides fixed200 scaled400
```

Results → `/tmp/pass5_results.json` + stdout.

## Related programs

- `pass4.py` — model builders / device / BASE_MODEL / warm (shared infra)
- `regime_forecast.py` — regime-conditioned extension of this harness
- `pass5_sweep.py` — systematic sweep over windows/strides/steps/tickers
  (resumable with `--resume`, bounded with `--max-experiments`)
- `granite_backfill.py` — data cleaning + Granite config
