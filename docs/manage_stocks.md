# manage_stocks.py

Maintain `monitored_stocks.parquet` membership, sectors, and flags (`index_member`, `in_portfolio`, `defensive_value_index`).

## Purpose
Single registry of tickers for fertilizer index, defensive value index, portfolio, and analytics filters.

## Typical operations
- Add/remove tickers and sectors
- Toggle index membership and portfolio flags
- List by status (`active` / `monitored`)

Used by almost every downstream script via ticker resolution.

## Related programs

- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/build_index.md](build_index.md)
- [docs/build_growth_tech_index.md](build_growth_tech_index.md)
- [docs/run_daily_automation.md](run_daily_automation.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
