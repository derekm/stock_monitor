# risk_metrics_ext.py

risk_metrics_ext.py — Liquidity, concentration, factor-style risk (Polars + pandas).

## Why it exists (rationale)

Liquidity, concentration, factor-style risk metrics (Polars + pandas) extending the risk picture beyond `portfolio_optimization`.

## Usage

```bash
python risk_metrics_ext.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `portfolio_holdings.parquet`
- **Screen / decision** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `portfolio_risk_summary.csv`
  - `risk_metrics_ext.csv`
- **Summary / metrics** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `preferred_metrics.csv`


## Related programs

- [docs/risk_enrich.md](risk_enrich.md)
- [docs/portfolio_optimization.md](portfolio_optimization.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
