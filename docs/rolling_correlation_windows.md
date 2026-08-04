# rolling_correlation_windows.py

rolling_correlation_windows.py — Rolling pairwise & sector correlation windows.

## Why it exists (rationale)

Rolling pairwise + sector correlation windows — finer time resolution than `allpairs_correlations`; feeds `maintain_analytics`.

## Usage

```bash
python rolling_correlation_windows.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `monitored_stocks.parquet`
- **Correlation matrix** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `rolling_corr_avg_timeseries.csv`
  - `rolling_corr_stability_by_asset.csv`
  - `rolling_sector_corr_windows.csv`


## Related programs

- [docs/allpairs_correlations.md](allpairs_correlations.md)
- [docs/cross_asset_analysis.md](cross_asset_analysis.md)
- [docs/maintain_analytics.md](maintain_analytics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
