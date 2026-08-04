# window_padding.py

window_padding.py — fill a sub-512 context for short-history tickers.

## Why it exists (rationale)

Fills a sub-512 context window for short-history tickers so TTM models have enough lookback — supports `ttm_features` / `granite_daily`.

## Usage

```bash
python window_padding.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- No persistent output files (in-memory / prints to stdout, or writes to a base parquet table listed in [docs/SCHEMAS.md](SCHEMAS.md)).


## Related programs

- [docs/ttm_features.md](ttm_features.md)
- [docs/granite_daily.md](granite_daily.md)
- [docs/backfill_historical.md](backfill_historical.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
