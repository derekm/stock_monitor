# sp_history_simulation.py

sp_history_simulation.py — reproduce S&P 500 inclusion/exclusion decisions in our independent simulation, and track our reimplementation vs the actuals.

## Why it exists (rationale)

Reproduces S&P 500 inclusion/exclusion decisions in our independent simulation and tracks reimplementation vs actuals.

## Usage

```bash
python sp_history_simulation.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/sp_index_methodology.md](sp_index_methodology.md)
- [docs/sp_universe_tracking.md](sp_universe_tracking.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
