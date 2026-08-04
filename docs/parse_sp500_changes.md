# parse_sp500_changes.py

parse_sp500_changes.py — build the authoritative S&P 500 ADD/REMOVE event log.

## Why it exists (rationale)

Builds the authoritative S&P 500 ADD/REMOVE event log from change pages — the event source for `sp_index_methodology` and `reconcile_sp500`.

## Usage

```bash
python parse_sp500_changes.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/parse_sp500.md](parse_sp500.md)
- [docs/sp_index_methodology.md](sp_index_methodology.md)
- `parse_tickerleague_changes.py`
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
