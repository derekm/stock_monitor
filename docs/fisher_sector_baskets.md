# fisher_sector_baskets.py

fisher_sector_baskets.py — Fisher-style price indexes for sector baskets inside an index sleeve.

## Why it exists (rationale)

Computes Fisher price indexes for sector baskets inside an index sleeve — finer-grained quantity/price decomposition than the whole-index `fisher_index` / `run_fisher_duckdb`.

## Usage

```bash
python fisher_sector_baskets.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Auxiliary table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `fisher_sector_baskets.csv`
  - `fisher_sector_baskets_latest.csv`
- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `monitored_stocks.parquet`


## Related programs

- [docs/fisher_index.md](fisher_index.md)
- [docs/run_fisher_duckdb.md](run_fisher_duckdb.md)
- [docs/build_index.md](build_index.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
