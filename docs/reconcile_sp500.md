# reconcile_sp500.py

Reconcile the `sp500_member` / `sp500_sector` / `sp500_date_added` columns in
`monitored_stocks.parquet` against the authoritative `sp500_constituents.parquet`.

## Why it exists (rationale)

The authoritative S&P 500 list is the Wikipedia-derived current constituents
(U.S.-listed common stocks only). `stock_monitor` (built by a prior assistant)
incorrectly carried ADRs/ETFs and stale flags into `sp500_member`. This script
overwrites those three columns from the authoritative source so the S&P-tracking
subsystem is correct.

## Usage

```bash
python reconcile_sp500.py
```

Flags: none. Overwrites `monitored_stocks.parquet` in place (the three S&P
columns only).

## Outputs

- `monitored_stocks.parquet` — `sp500_member` / `sp500_sector` / `sp500_date_added`
  corrected in place

(Schema family: aux_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [parse_sp500.md](parse_sp500.md) — the authoritative list
- [sp_universe_tracking.md](sp_universe_tracking.md) / [sp_index_methodology.md](sp_index_methodology.md)
- [manage_stocks.md](manage_stocks.md)
