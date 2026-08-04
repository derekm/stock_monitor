# cross_asset_analysis.py

Cross-asset and cross-sector correlation analysis; builds sector EW price
series for forecasting.

## Why it exists (rationale)

Diversification decisions need to know how sectors and single names relate, and
the forecasting exogenous features need sector EW return series. This script
produces the sector correlation matrices, asset-to-sector betas, and rolling
cross-asset correlations, and writes `sector_prices.parquet` that
`ttm_exogenous.py` consumes.

## Usage

```bash
python cross_asset_analysis.py all
python cross_asset_analysis.py sectors
python cross_asset_analysis.py assets --tickers MOS,CF,SHEL,BAYRY
python cross_asset_analysis.py rolling --window 20
python cross_asset_analysis.py save-sector-prices
```

Subcommands: `all`, `sectors`, `assets`, `rolling`, `save-sector-prices`.

## Outputs

- `sector_correlation_matrix.csv` — sector × sector correlation
- `asset_sector_correlations.csv` — asset-to-sector corr/beta
- `rolling_cross_asset_correlations.csv` — rolling key-pair correlations
- `cross_asset_stability.csv` — correlation stability metrics
- `sector_prices.parquet` / `sector_tickers.csv` — `SECT_*` synthetic tickers
  (used by `ttm_exogenous.py`)

(Schema families: correlation_matrix / aux_table / base_table — see
[SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [ttm_exogenous.md](ttm_exogenous.md) — consumes `sector_prices.parquet`
- [allpairs_correlations.md](allpairs_correlations.md) / [crisis_correlation.md](crisis_correlation.md)
- [maintain_analytics.md](maintain_analytics.md)
