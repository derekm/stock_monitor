# pass3_sweep.py

pass3_sweep.py - Granite TTM parameter sweep (Pass 3).

## Why it exists (rationale)

Granite TTM parameter sweep (Pass 3) — experiments feeding `granite_backfill` / `ttm_backfill` config choices.

## Usage

```bash
python pass3_sweep.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/granite_backfill.md](granite_backfill.md)
- [docs/ttm_backfill.md](ttm_backfill.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
