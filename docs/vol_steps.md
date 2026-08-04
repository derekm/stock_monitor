# vol_steps.py

vol_steps.py - Volatility-based per-ticker training-step allocator.

## Why it exists (rationale)

Volatility-based per-ticker training-step allocator for TTM backfill.

## Usage

```bash
python vol_steps.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/ttm_backfill.md](ttm_backfill.md)
- [docs/granite_backfill.md](granite_backfill.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
