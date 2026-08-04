# live_index_backtest.py

Parameterized index / sleeve backtest with Sharpe comparison.

## Why it exists (rationale)

Parameterized index/sleeve backtest with Sharpe comparison — validates `build_index` / `build_growth_tech_index` / `build_defensive_index` constructions vs benchmarks.

## Usage

```bash
python live_index_backtest.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Screen / decision** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `index_backtest_stats.csv`
- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `index_levels_1y.csv`
  - `index_levels_1y.parquet`
- **Summary / metrics** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `sharpe_comparison.csv`


## Related programs

- [docs/build_index.md](build_index.md)
- [docs/build_growth_tech_index.md](build_growth_tech_index.md)
- [docs/build_defensive_index.md](build_defensive_index.md)
- [docs/sharpe_comparison.md](sharpe_comparison.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
