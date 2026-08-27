# ttm_features.py

Build TTM-ready multivariate panels from `daily_prices/`.

## Why it exists (rationale)

Granite TTM needs consistent business-day panels with multiple channels (price,
volume, returns, volatility, simple indicators). This builds those panels per
ticker/index (`close_only` or `full` channel mode) as parquet that `ttm_backfill`
trains on and `forecast_granite` / `granite_daily` read at forecast time.

## Usage

```bash
python ttm_features.py --index portfolio --save
python ttm_features.py --index portfolio --mode full --save
python ttm_features.py --tickers AEP,NVR --mode close_only
```

Flags (via `cli_common` + own): `--index/--universe`, `--ticker`, `--mode`
(`close_only` default / `full`), `--save`. Reads `daily_prices/`,
`monitored_stocks.parquet`.

## Outputs

- `ttm_panels/<index>.parquet` (or per-ticker panels) — multivariate TTM panels
  (path set around line 190)

(Schema family: base_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [ttm_backfill.md](ttm_backfill.md) — trains on these panels
- [ttm_exogenous.md](ttm_exogenous.md) — exogenous channels
- [forecast_granite.md](forecast_granite.md) / [granite_daily.md](granite_daily.md)
- [backfill_historical.md](backfill_historical.md) — history source
