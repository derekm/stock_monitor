# signal_model.py

Supervised (sklearn GradientBoosting) blend of the five signal families,
validated with cross-sectional K-fold against the IC-weighted composite.

## Why it exists (rationale)

Closes the "signal combination — supervised ML" TODO. `signal_aggregator.py`
is a linear, OOS-IC-weighted combination; this adds a nonlinear path that can
capture interactions between families (e.g., peer + earnings together).

## Method & honesty

- Features: latest signal snapshot per family (missing → neutral 0.5).
- Target: forward 21d return measured at cutoff − 21d (the same
  no-future-leak convention as the aggregator — the target is observable at
  the live point).
- Validation: 4-fold cross-sectional K-fold (shuffled). The dataset is a
  single-date cross-section, so there is no temporal ordering to purge —
  K-fold tests generalization across NAMES only; the temporal leak guard is
  upstream in the target construction.
- Reported honestly: mean OOS rank IC of the model vs the equal-weight
  composite — the supervised path must justify itself against the simpler
  baseline.

## Usage

```bash
python signal_model.py --save
```

## Outputs

- `signal_model_oos.csv` — per-fold (model_ic, composite_ic, ic_delta)
- `signal_model_weights.csv` — feature importances

## Related programs

- `signal_aggregator.py` — the linear baseline this compares against
- `cv_utils.py` — the honest-OOS discipline
