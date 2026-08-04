# polars_export_utils.py

Helpers for streaming large parquet → JSON records without full pandas copies.
This is a **library**, not a runnable script.

## Why it exists (rationale)

The dashboard's export path must turn big parquet tables (e.g. years of
`daily_prices`) into JSON records for DuckDB-Wasm without materializing a giant
pandas DataFrame. These helpers use Polars lazy scanning + tail-windowing so only
the needed rows are collected.

## Key functions

- `tail_prices_records(path, tickers=None, days=420, limit=80000)` — recent price
  rows as records (optionally filtered by ticker), capped.
- (related helpers for the export pipeline — see source for the full set.)

## Outputs

None (library; returns `list[dict]`).

## Related programs

- [export_dashboard_data.md](export_dashboard_data.md) — uses these for large tables
- [data_access.md](data_access.md) — pandas-based loaders
