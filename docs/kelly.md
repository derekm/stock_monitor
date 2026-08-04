# kelly.py

Estimate Kelly criterion position fractions from stored μ/σ ranges (`kelly_parameters.parquet`).

## Purpose
Map defensive names (PG, JNJ, KO, …) and active ideas to fractional Kelly (½ / ¼) sizing suggestions.

## Formula (continuous)
`f* = (μ − r) / σ²` then apply fractional Kelly and portfolio caps.

Combine with valuation screens; Kelly is a sizing aid, not a buy signal alone.

## Related programs

- [docs/vol_target.md](vol_target.md)
- [docs/portfolio_optimization.md](portfolio_optimization.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
