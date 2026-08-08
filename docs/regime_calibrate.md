# regime_calibrate.py — calibration + coverage training for regime models

Two production-hardening jobs for the regime-selected Granite-TTM serving:

## 1. Calibration check (default)

Verifies the MC-dropout z=1 std band is honest: run the (fine-tuned) model's
OOS test windows through `forecast_ttm_mc_dropout`, then measure how often
the realized price lands inside mean ± 1σ. Coverage ≈ 68% is well-calibrated;
far off means the band is lying (over-confident if < 68%, under-confident if
> 68%).

Output: `regime_calibration.csv` — (ticker, regime, n_test_windows,
mc_band_cov_z1, expected_if_calibrated) + a verdict line.

## 2. Coverage training (--train)

Trains regime checkpoints for tickers that have no pass6 coverage, so regime
selection extends beyond AEP/NVR/FICO. One model per (ticker, current
regime) at the pass6-most-common best config (steps=3000, cap=100, lr=None)
from pass7's matrix. Checkpoints land in `checkpoints/regime/` and are
picked up automatically by `forecast_granite.py` serving.

## Usage

```bash
python regime_calibrate.py --tickers AEP,NVR,FICO          # calibration
python regime_calibrate.py --tickers MSFT,GOOG,JPM --train # coverage
```

## Related programs

- `forecast_granite.py` — `forecast_ttm_mc_dropout` (the band being checked)
- `regime_serving.py` — the serving path these checkpoints feed
- `pass6.py` — the training harness reused here
