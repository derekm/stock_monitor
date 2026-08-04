# bench_backfill.py

Device-throughput benchmark: CPU vs MX550 for tiny-model (TTM-class, ~1M param) rolling-window time-series training. Isolates whether the MX550 gives a speedup for our small-batch backfill workload. Uses a generic small model so it doesn't 

## Why it exists (rationale)

Device-throughput benchmark (CPU vs MX550) for tiny TTM-class rolling-window training — validates the backfill hardware strategy.

## Usage

```bash
python bench_backfill.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/backfill_historical.md](backfill_historical.md)
- [docs/ttm_backfill.md](ttm_backfill.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
