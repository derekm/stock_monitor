# ttm_features.py

Build TTM-ready multivariate panels from `daily_prices.parquet`.

## Channels
close, volume, log returns, 20d vol, RSI-14, MA ratio, HL range, volume z-score

## Usage
```bash
python ttm_features.py --index portfolio --mode close_only --save
python ttm_features.py --ticker MOS --save
```

Panels land under `ttm_panels/` for Granite multivariate forecasting.

## Related programs

- [docs/backfill_historical.md](backfill_historical.md)
- [docs/update_prices.md](update_prices.md)
- [docs/ttm_exogenous.md](ttm_exogenous.md)
- [docs/forecast_granite.md](forecast_granite.md)
- [docs/granite_daily.md](granite_daily.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
