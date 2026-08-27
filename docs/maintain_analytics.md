# maintain_analytics.py

Regenerate all analysis CSV files from the parquet sources in one pass.

## Why it exists (rationale)

After a data refresh (`update_prices`, `update_fundamentals`, `backfill_*`),
every downstream CSV is stale. This is the "rebuild everything" orchestrator: it
re-runs the correlation, regime, risk, screen, index, and forecast analytics and
rewrites their CSVs. `run_daily_automation` and the dashboard's refresh both call
it (or its sub-commands).

## Usage

```bash
python maintain_analytics.py all
python maintain_analytics.py correlations
python maintain_analytics.py optimize
python maintain_analytics.py screens
```

Sub-commands (from its `main`): `all`, `correlations`, `regimes`, `risk`,
`screens`, `indexes`, `optimize`, `forecasts`, and more. Reads
`daily_prices/`, `monitored_stocks.parquet`, `portfolio_holdings.parquet`,
`index_levels_1y.parquet`.

## Outputs (selected)

- `sector_correlation_matrix.csv`, `fertilizer_correlation_matrix.csv`,
  `rolling_sector_correlations.csv`, `correlation_stability_metrics.csv`
- `hmm_2state_regimes.csv`, `hmm_2state_regime_correlations.csv`,
  `kalman_correlations.csv`
- …and the per-analytic CSVs of the scripts it invokes (see their docs)

(Schema family: depends on sub-command — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [run_daily_automation.md](run_daily_automation.md) — calls this
- Every analytics script it wraps (correlations, regimes, risk, screens, indexes)
- [export_dashboard_data.md](export_dashboard_data.md) — ships the result
