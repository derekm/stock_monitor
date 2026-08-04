# sp_index_methodology.py

sp_index_methodology.py — S&P 500 inclusion/exclusion reimplementation + our dual-pass strength tiers, tracked against S&P actuals.

## Why it exists (rationale)

Reimplements S&P 500 inclusion/exclusion logic with our dual-pass strength tiers, scored against S&P actuals (the S&P-ACTUALS tracking system).

## Usage

```bash
python sp_index_methodology.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/sp_universe_tracking.md](sp_universe_tracking.md)
- [docs/sp_history_simulation.md](sp_history_simulation.md)
- [docs/parse_sp500_changes.md](parse_sp500_changes.md)
- [docs/reconcile_sp500.md](reconcile_sp500.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
