# pass7.py — experiment-design matrix for regime-selected models

Tests whether pass6's per-regime best-config findings are ROBUST across
experiment designs — the "several different experiment designs with different
mixes" idea — and whether regime-aware training scheduling helps.

## Arms (all with pass6 honesty rules: shared global boundary, 96d embargo,
per-regime persistence baseline, IBM-base-only fine-tuning)

- **boundary** — split_frac ∈ {0.55, 0.70, 0.85}: does the best config
  survive a different train/test boundary year?
- **composition** — pure (train only on the regime's windows) vs all (train
  on every window, evaluate per regime): does regime specialization beat
  more training data?
- **lr** — gd.LR (1e-4) vs 5e-5: is the finding lr-sensitive?
- **freshness** — full in-regime history vs only the most recent ~10y of
  in-regime windows: the "hold off trainings until the trend switches back
  into that model's regime" test.

Every unique (split_frac, steps, cap, lr, composition) cell is trained ONCE
and tagged with every arm it belongs to (no duplicate training).

## Outputs

- `/tmp/pass7_results.jsonl` — append-only, resumable (`--resume`)
- `regime_model_matrix.csv` — every cell result (arm, ticker, regime,
  split_frac, steps, cap, lr, composition, dir_acc, pers_dir, mape)
- `regime_model_matrix_summary.csv` — per-arm mean/max OOS dir excess +
  most-common best config

## Usage

```bash
python pass7.py --tickers AEP,NVR --arms boundary composition lr freshness
python pass7.py --resume --max-experiments 20
python pass7.py --quick
```

## Related programs

- `pass6.py` — the single-design pass this generalizes (shared machinery)
- `pass5.py` / `regime_forecast.py` — the honest-OOS harness
