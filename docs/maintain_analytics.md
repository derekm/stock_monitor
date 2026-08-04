# maintain_analytics.py

CLI hub to regenerate analytics CSVs: correlations, rolling/stability, HMM regimes, Kalman tracking, VAR/Granger, index backtests, cross-asset suite.

## Usage
```bash
python maintain_analytics.py all
python maintain_analytics.py correlations
python maintain_analytics.py rolling
python maintain_analytics.py stability
python maintain_analytics.py hmm
python maintain_analytics.py kalman
python maintain_analytics.py var
python maintain_analytics.py backtest
python maintain_analytics.py cross-asset
python maintain_analytics.py growth-tech
python maintain_analytics.py optimize
python maintain_analytics.py vol-rp
python maintain_analytics.py list
```

## Outputs (examples)
- `sector_correlation_matrix.csv`
- `fertilizer_correlation_matrix.csv`
- `rolling_sector_correlations.csv`
- `correlation_stability_metrics.csv`
- `hmm_2state_regimes.csv` / `hmm_2state_regime_correlations.csv`
- `kalman_correlations.csv`
- `granger_causality_sectors.csv`
- `index_backtest_stats.csv`

Dashboard **CSV Catalog** runs SQL that reproduces these tables from embedded data.
