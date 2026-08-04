# rolling_window_analysis.py

rolling_window_analysis.py — Rolling vol, beta, Sharpe, max-DD, dual-screen stability.

## Why it exists (rationale)

Rolling vol, beta, Sharpe, max-DD, dual-screen stability — the time-varying risk/stability view behind `maintain_analytics`.

## Usage

```bash
python rolling_window_analysis.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `monitored_stocks.parquet`
  - `portfolio_holdings.parquet`
- **Screen / decision** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `rolling_screen_stability.csv`
- **Summary / metrics** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `rolling_window_metrics.csv`


## Related programs

- [docs/maintain_analytics.md](maintain_analytics.md)
- [docs/risk_metrics_ext.md](risk_metrics_ext.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
