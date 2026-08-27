# run_fisher_duckdb.py

Chained Fisher / Laspeyres / Paasche price & quantity indexes computed **in
DuckDB** — a reimplementation/check of `fisher_index.py`.

## Why it exists (rationale)

`fisher_index.py` (Pandas) is the reference. This recomputes the same chained
Fisher indexes directly in DuckDB (price = close, quantity = volume, ffill'd;
chained levels base = 100) so the S&P-tracking reconciliation and any
large-history runs get a fast, SQL-native path — and a cross-check that the two
implementations agree.

## Usage

```bash
python run_fisher_duckdb.py --universe portfolio --save
python run_fisher_duckdb.py --universe all --years 5
```

Flags (via `cli_common` + own): `--universe/--index`, `--ticker`, `--years`,
`--save`. Reads `daily_prices/`, `monitored_stocks.parquet`.

## Outputs

- `fisher_indexes_duckdb.parquet` — DuckDB-computed Fisher levels
- (and related CSV/parquet levels; see source)

(Schema family: index_levels — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [fisher_index.md](fisher_index.md) — the Pandas reference implementation
- [fisher_sector_baskets.md](fisher_sector_baskets.md)
- [sp_index_methodology.md](sp_index_methodology.md) / [sp_universe_tracking.md](sp_universe_tracking.md)
- [backfill_historical.md](backfill_historical.md)
