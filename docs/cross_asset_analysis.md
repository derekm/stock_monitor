# cross_asset_analysis.py

Cross-asset and **cross-sector** correlation analysis; builds sector EW price series for forecasting.

## Usage
```bash
python cross_asset_analysis.py all
python cross_asset_analysis.py sectors
python cross_asset_analysis.py assets --tickers MOS,CF,SHEL,BAYRY
python cross_asset_analysis.py rolling --window 20
python cross_asset_analysis.py save-sector-prices
```

## Outputs
- `sector_correlation_matrix.csv`
- `asset_sector_correlations.csv`
- `rolling_cross_asset_correlations.csv`
- `cross_asset_stability.csv`
- `sector_prices.parquet` / `sector_tickers.csv` (`SECT_*` synthetic tickers)

## Decision relevance
Materials vs Staples/Health Care correlations inform whether fertilizer/value sleeves diversify the portfolio.
