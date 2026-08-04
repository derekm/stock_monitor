# rebalance_calendar.py

rebalance_calendar.py — Regime- and dual-pass-aware rebalance schedule.

## Why it exists (rationale)

Regime- and dual-pass-aware rebalance schedule generator — turns screen/stress output into actionable rebalance dates.

## Usage

```bash
python rebalance_calendar.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
- **Regime / state table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_regimes.csv`
- **Summary / metrics** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `preferred_metrics.csv`
  - `rebalance_calendar.csv`


## Related programs

- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/stress_dual_pass.md](stress_dual_pass.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
