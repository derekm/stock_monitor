# portfolio_report.py

Report holdings, cost basis, unrealized P&L, weights, and sector mix from `trades.parquet` / `portfolio_holdings.parquet`.

## Purpose
Snapshot personal portfolio performance and composition for comparison to S&P-style benchmarks and index backtests.

## Inputs
- `trades.parquet` (Robinhood fills / DRIPs)
- `daily_prices.parquet` for last marks

## Related programs

- [docs/portfolio_optimization.md](portfolio_optimization.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/manage_stocks.md](manage_stocks.md)
- [docs/run_daily_automation.md](run_daily_automation.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
