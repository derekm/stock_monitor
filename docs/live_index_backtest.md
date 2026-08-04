# live_index_backtest.py

Parameterized index / sleeve backtest with Sharpe comparison.

## Why it exists (rationale)

The indexes built by `build_index` / `build_defensive_index` /
`build_growth_tech_index` need to be measured, not assumed. This backtests one or
more sleeves over a trailing window, compares Sharpe vs a benchmark (e.g. SPY),
and can emit a machine-readable JSON summary for the pipeline/dashboard.

## Usage

```bash
python live_index_backtest.py --years 1 --rf 0.04
python live_index_backtest.py --indexes fertilizer,defensive,portfolio,growth --years 2 --benchmark SPY
python live_index_backtest.py --years 1 --json
```

Flags: `--years`, `--rf` (risk-free, default 0.04), `--indexes` (comma list),
`--benchmark` (default SPY), `--json`.

## Outputs

- `index_backtest_stats.csv` — per-index backtest stats
- `index_levels_1y.parquet` — 1y index levels
- `sharpe_comparison.csv` — Sharpe vs benchmark

(Schema families: index_levels / weights_performance — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [build_index.md](build_index.md) / [build_defensive_index.md](build_defensive_index.md) / [build_growth_tech_index.md](build_growth_tech_index.md)
- [fisher_index.md](fisher_index.md) / [run_fisher_duckdb.md](run_fisher_duckdb.md)
- [maintain_analytics.md](maintain_analytics.md)
