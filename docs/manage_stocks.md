# manage_stocks.py

Maintain `monitored_stocks.parquet` membership, sectors, and flags (`index_member`, `in_portfolio`, `defensive_value_index`).

## Purpose
Single registry of tickers for fertilizer index, defensive value index, portfolio, and analytics filters.

## Typical operations
- Add/remove tickers and sectors
- Toggle index membership and portfolio flags
- List by status (`active` / `monitored`)

Used by almost every downstream script via ticker resolution.
