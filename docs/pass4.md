# pass4.py

pass4.py - Pass-4: re-run passes 2 & 3 on ADJUSTED closes, warm-started from the freshly-trained ADJUSTED global checkpoint (train_adjusted_full.py output).

## Why it exists (rationale)

Pass-4 re-runs passes 2&3 on ADJUSTED closes warm-started from the adjusted global checkpoint — measures relative param signal from undertrained checkpoints.

## Usage

```bash
python pass4.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/train_adjusted_full.md](train_adjusted_full.md)
- [docs/granite_backfill.md](granite_backfill.md)
- [docs/ttm_backfill.md](ttm_backfill.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
