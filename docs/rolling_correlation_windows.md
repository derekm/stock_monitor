# rolling_correlation_windows.py

Rolling pairwise & sector correlation windows.

## Why it exists (rationale)

Correlation is not static. This builds rolling pairwise and sector correlation
series so the dashboard can show correlation creeping up (a diversification
warning) and feed `regime_correlation_breakdown` / `crisis_correlation`. It
complements `allpairs_correlations` (which gives full-history + latest matrices).

## Usage

```bash
python rolling_correlation_windows.py --save
```

Flags: `--save`. Reads `daily_prices.parquet`, `monitored_stocks.parquet`.

## Outputs

- `rolling_corr_avg_timeseries.csv` — avg/median pairwise corr over time
- `rolling_corr_sector.csv` — sector correlation over time

(Schema family: correlation_matrix — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [allpairs_correlations.md](allpairs_correlations.md)
- [crisis_correlation.md](crisis_correlation.md)
- [regime_correlation_breakdown.md](regime_correlation_breakdown.md)
