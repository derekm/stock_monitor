# parse_sp500_changes.py

Build the authoritative S&P 500 ADD/REMOVE event log: `sp500_changes.parquet`.

## Why it exists (rationale)

Index-replication and event studies need the *history* of constituents, not just
the current list. This pulls the official S&P announcements from
tickerleague.com (a JSON array embedded in the page, client-paginated back to
1957 — 1,500+ real events) and falls back to the Wikipedia "List of S&P 500
companies" changes table (1976–2026) only if tickerleague is unreachable. It is
the event source for `sp_universe_tracking` / `sp_index_methodology`.

## Usage

```bash
python parse_sp500_changes.py
```

Flags: none. Output is the parquet of (ticker, event_date, action, …).

## Outputs

- `sp500_changes.parquet` — ADD/REMOVE events, ~1957→present

(Schema family: aux_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [parse_sp500.md](parse_sp500.md) — current constituents
- [parse_tickerleague_changes.md](parse_tickerleague_changes.md) — the primary source scraper
- [sp_universe_tracking.md](sp_universe_tracking.md) / [sp_index_methodology.md](sp_index_methodology.md)
- [run_fisher_duckdb.md](run_fisher_duckdb.md)
