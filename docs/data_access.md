# data_access.py

Shared loaders for the parquet/CSV tables used across stock_monitor programs.
This is a **library**, not a runnable script (no argparse, no main).

## Why it exists (rationale)

Every analytics program reads the same base tables (`monitored_stocks`,
`daily_prices`, `portfolio_holdings`, `trades`, `fundamentals`,
`sector_prices`). Centralizing the loaders avoids N copy-paste read paths, keeps
column types consistent (dates parsed, fundamentals reduced to latest snapshot),
and gives a single place to handle fallback locations (e.g. `trades.parquet`
also checked in the parent dir).

## Key functions

| Function | Returns | Notes |
|----------|---------|-------|
| `load_stocks()` | `monitored_stocks.parquet` | empty frame if missing |
| `load_prices(tickers=None, columns=None)` | `daily_prices/` | dates parsed; optional ticker/column filter |
| `load_holdings()` | `portfolio_holdings.parquet` | |
| `load_trades()` | `trades.parquet` | also checks parent dir; parses `filled_datetime` |
| `load_fundamentals(latest=True)` | `fundamentals.parquet` | `latest=True` → one row per ticker (latest `as_of_date`) |
| `price_matrix(tickers=None, field="close")` | wide date×ticker DataFrame | pivot + forward-fill |

## Outputs

None (library).

## Related programs

- [cli_common.md](cli_common.md) — flag parsing / ticker resolution
- [data_integrity.md](data_integrity.md) / [data_integrity_deep.md](data_integrity_deep.md)
  — validation that uses these loaders
- Any program doc that lists `daily_prices/` / `fundamentals.parquet` as input
