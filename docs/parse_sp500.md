# parse_sp500.py

Parse the S&P 500 constituents table from the downloaded Wikipedia HTML into
`sp500_constituents.parquet` — the authoritative constituent list.

## Why it exists (rationale)

The S&P-tracking subsystem needs a clean, current constituent list. Wikipedia's
table has an inconsistent layout (some rows carry an extra "Headquarters"
column), so this parser locates the first wikitable, matches known header texts,
and maps each data row by **position** against that header (skipping unknown
columns). It is the input to `backfill_constituents` and `sp_universe_tracking`.

## Usage

```bash
python parse_sp500.py          # reads the local Wikipedia HTML, writes parquet
```

Flags: none. Output is written via DuckDB `COPY ... PARQUET`.

## Outputs

- `sp500_constituents.parquet` — ticker + metadata, ordered by ticker

(Schema family: aux_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [parse_sp500_changes.md](parse_sp500_changes.md) — ADD/REMOVE event log
- [backfill_constituents.md](backfill_constituents.md) — fills fundamentals for these
- [sp_universe_tracking.md](sp_universe_tracking.md) / [sp_index_methodology.md](sp_index_methodology.md)
- [run_fisher_duckdb.md](run_fisher_duckdb.md)
