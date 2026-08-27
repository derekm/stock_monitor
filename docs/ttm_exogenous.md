# ttm_exogenous.py

Build exogenous feature channels for Granite TTM forecasts.

## Why it exists (rationale)

Granite TTM multivariate forecasts are stronger with market/sector context. This
builds the exogenous panel from local parquet (no network): equal-weight market
return of all monitored names, sector equal-weight returns, cross-sectional
dispersion, and optional external CSV — written as `exogenous_panel.parquet` that
`forecast_granite` / `ttm_features` consume as extra channels.

## Usage

```bash
python ttm_exogenous.py --save
python ttm_exogenous.py --from-csv extra.csv --save
```

Flags: `--from-csv` (optional external CSV with a date column), `--save`. Reads
`daily_prices/`, `monitored_stocks.parquet`.

## Outputs

- `exogenous_panel.parquet` — per-date exogenous channels
  (market return, sector returns, cross-sec dispersion)

(Schema family: base_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [ttm_features.md](ttm_features.md) — combines with price panel
- [forecast_granite.md](forecast_granite.md) — uses the channels
- [cross_asset_analysis.md](cross_asset_analysis.md) — sector EW prices source
- [tspulse_anomaly.md](tspulse_anomaly.md) — also reads the panel
