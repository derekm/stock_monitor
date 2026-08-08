# regime_serving.py — serve regime-selected Granite-TTM checkpoints

Bridges pass6/pass7 research (per-regime fine-tuned models, validated OOS)
into forecast_granite's daily serving path. For each ticker:

1. Read `regime_model_best.csv` for the CURRENT HMM regime (the pass6
   best-config selection: max OOS direction excess over the regime's
   persistence baseline).
2. If a checkpoint exists under `checkpoints/regime/` for (ticker, regime),
   return it — the forecast is regime-SELECTED.
3. Otherwise return None — the caller uses the general model, with a
   reason (no_regime / no_coverage / no_checkpoint).

Per-(ticker, current-regime) selection. Tickers without pass6 coverage keep
the general model — regime selection is an upgrade when available, never a
downgrade.

## Checkpoint format

Saved by `pass6.py --ckpt-dir checkpoints/regime` as
`<TICKER>__<regime>__<steps>__<lr>.pt`, containing `model` (state dict),
`dir_acc`, `n_channels` (1 = close-only, 3 = close+return+vol20), `tag`,
`trained_on` (ISO timestamp — drives the staleness flag).

## Staleness

`serve_regime_model` returns `age_days` (from `trained_on`, falling back to
file mtime). `forecast_granite.py` warns when a served checkpoint is older
than 90 days — the retrain schedule: re-run pass6 when the regime's windows
have refreshed.

## Functions

- `current_regime()` — latest HMM regime label
- `best_config_for(ticker, regime)` — pass6 best config + per-span dir_acc
- `serve_regime_model(ticker)` — (ckpt_path, cfg, reason)
- `serving_report(tickers)` — table for humans

## Related programs

- `pass6.py` — trains and saves the checkpoints
- `forecast_granite.py` — consumes serving (ensemble: general + regime model)
- `regime_calibrate.py` — calibration check + coverage training
