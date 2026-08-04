# granite_daily.py

granite_daily.py — Daily 512-day -> 96-day Granite TTM forecast + CONTINUAL RETRAINING on prior-day actuals.

## Why it exists (rationale)

Production daily forecaster: 512→96-day Granite TTM run with continual retraining on prior-day actuals — the live engine behind `granite_service` and `forecast_granite`.

## Usage

```bash
python granite_daily.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Forecast / anomaly** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `forecasts_granite.parquet`
- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `granite_series_cache.parquet`


## Related programs

- [docs/forecast_granite.md](forecast_granite.md)
- [docs/granite_backfill.md](granite_backfill.md)
- [docs/ttm_backfill.md](ttm_backfill.md)
- [docs/granite_service.md](granite_service.md)
- [docs/train_adjusted_full.md](train_adjusted_full.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
