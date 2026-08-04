# growth_tech_analytics.py

Full analysis suite for the higher-risk growth/tech index (the 4th sleeve),
mirroring the sector/portfolio analytics.

## Why it exists (rationale)

The growth/tech sleeve gets the same analytic treatment as the core book so it
can be compared on vol, correlation, risk-model behavior, and backtest vs the
defensive / fertilizer / portfolio proxies — and so its per-name caps are
informed by real risk, not guesses.

## Usage

```bash
python growth_tech_analytics.py --save
python growth_tech_analytics.py --universe growth_tech
```

Flags (via `cli_common`): `--universe/--index`, `--ticker`, `--save`. Reads
`daily_prices.parquet`, `monitored_stocks.parquet`, `portfolio_holdings.parquet`,
`growth_tech_index_levels.parquet`.

## Outputs

- `growth_tech_membership.csv` — current sleeve membership
- `growth_tech_vol_returns.csv` — realized vol / return summary
- `growth_tech_correlation_matrix.csv` / `growth_tech_sleeve_correlations.csv`
- `growth_tech_rolling_corr.csv` / `growth_tech_corr_stability.csv`
- `growth_tech_index_levels_compare.parquet` / `.csv` — vs other sleeves
- `growth_tech_backtest_stats.csv`
- `growth_tech_risk_models.csv` — ERC / InvVol / GMV / vol-target comparison
- `growth_tech_sleeve_performance.csv`

(Schema families: screen_decision / correlation_matrix / index_levels /
weights_performance — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [build_growth_tech_index.md](build_growth_tech_index.md) — the index it analyzes
- [factor_rotation_defense.md](factor_rotation_defense.md)
- [risk_parity_analytics.md](risk_parity_analytics.md) — growth_ai table
- [growth_tech_index.md](growth_tech_index.md) (if present)
