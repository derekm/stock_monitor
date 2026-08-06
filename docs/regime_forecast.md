# regime_forecast.py

Regime-conditioned Granite-TTM evaluation: does direction accuracy hold in
`high_vol_stress`, or should forecast trust be regime-gated?

## Why it exists (rationale)

`pass5.py` established Granite-TTM is a direction forecaster (beats
persistence on direction, loses on MAPE). The unanswered question: **is that
edge regime-dependent?** If the model only beats persistence in calm regimes,
forecasts in `high_vol_stress` should be down-weighted. This script answers it
with the same honest-OOS machinery (trainlast: train last 10y, test the 10y
preceding — disjoint).

## Method

- Test windows are tagged with the HMM regime in force during the ~20 trading
  days before the forecast point (majority of `hmm_regime_states.csv` labels
  at-or-before each window's context end) — the regime the forecaster was "in".
- Reports overall direction accuracy (model vs persistence) plus the test-set
  regime mix and the per-regime persistence baseline (fraction of windows
  where the realized forward move was up). The mix + per-regime persistence
  answer the data question: is directional predictability regime-dependent?

## Usage

```bash
python regime_forecast.py --tickers AEP KO XOM --steps 6000
python regime_forecast.py --steps 9000 --stride scaled400
python regime_forecast.py --mode half --cutoff-frac 0.5
```

## Outputs

- `regime_forecast_stats.csv` — per ticker: `dir_acc`, `mape`, `mape_pers`,
  `test_regime_mix` (window counts by regime), `persistence_dir_acc_by_regime`

## Related programs

- `pass5.py` — the honest-OOS harness this extends (same training protocol)
- `hmm_regime_detection.py` — produces the regime labels it consumes
- `pass5_sweep.py` — the systematic sweep this complements
