# sp_universe_tracking.py

sp_universe_tracking.py — track ALL S&P 500 constituents (503) by index, basket, and vertical, with our scored inclusion tiers where fundamentals exist.

## Why it exists (rationale)

Tracks all 503 S&P 500 constituents by index/basket/vertical with scored inclusion tiers where fundamentals exist.

## Usage

```bash
python sp_universe_tracking.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Other** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `sp500_universe_tracking.parquet`


## Related programs

- [docs/sp_index_methodology.md](sp_index_methodology.md)
- [docs/parse_sp500.md](parse_sp500.md)
- [docs/reconcile_sp500.md](reconcile_sp500.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
