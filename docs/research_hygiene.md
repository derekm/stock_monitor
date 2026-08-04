# research_hygiene.py

research_hygiene.py — Walk-forward inclusion rules + forecast reliability report.

## Why it exists (rationale)

Walk-forward inclusion rules + forecast reliability report — guards against look-ahead and over-fit screens; complements `preferred_metrics`.

## Usage

```bash
python research_hygiene.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Forecast / anomaly** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `forecast_backtest_metrics.csv`
  - `forecast_reliability_report.csv`
- **Screen / decision** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `inclusion_walkforward.csv`
- **Summary / metrics** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `preferred_metrics.csv`
  - `preferred_metrics_history.csv`


## Related programs

- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/forecast_reliability.md](forecast_reliability.md)
- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
