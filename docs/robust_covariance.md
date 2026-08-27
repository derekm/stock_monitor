# robust_covariance.py

Robust covariance estimators for portfolio optimization.

## Why it exists (rationale)

Sample covariance is noisy and unstable for optimization. This offers
alternatives (sample, Ledoit-Wolf shrinkage, exponentially-weighted, possibly
OAS) and compares them, so `portfolio_optimization` / `risk_parity_analytics`
can use a steadier Σ. It is the covariance-quality layer under the optimizers.

## Usage

```bash
python robust_covariance.py --save
python robust_covariance.py --universe portfolio
```

Flags (via `cli_common`): `--universe/--index`, `--ticker`, `--save`. Reads
`daily_prices/`, `portfolio_holdings.parquet`, `monitored_stocks.parquet`.

## Outputs

- `robust_covariance_summary.csv` — estimator comparison (condition number,
  eigenvalue spread, etc.)

(Schema family: weights_performance — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [portfolio_optimization.md](portfolio_optimization.md)
- [risk_parity_analytics.md](risk_parity_analytics.md)
- [black_litterman.md](black_litterman.md)
