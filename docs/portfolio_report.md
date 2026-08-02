# portfolio_report.py

Report holdings, cost basis, unrealized P&L, weights, and sector mix from `trades.parquet` / `portfolio_holdings.parquet`.

## Purpose
Snapshot personal portfolio performance and composition for comparison to S&P-style benchmarks and index backtests.

## Inputs
- `trades.parquet` (Robinhood fills / DRIPs)
- `daily_prices.parquet` for last marks
