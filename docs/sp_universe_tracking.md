# sp_universe_tracking.py

Track ALL S&P 500 constituents (503) by index, basket, and vertical, with our
scored inclusion tiers where fundamentals exist.

## Why it exists (rationale)

Fulfills "track all indexes and their baskets": every current constituent is
represented with its GICS sector / sub-industry (the basket/vertical) and, for
the ~28 names we carry fundamentals for, a scored inclusion tier. It is the
breadth view of the S&P-tracking subsystem — coverage, not just the personal
book.

## Usage

```bash
python sp_universe_tracking.py --save
```

Flags: `--save`. Reads `sp500_constituents.parquet`, `sp500_changes.parquet`,
`monitored_stocks.parquet`, `fundamentals.parquet`, `daily_prices.parquet`.

## Outputs

- `sp500_universe_tracking.parquet` — per-constituent tracking (sector,
  sub-industry, scored tier where available)

(Schema family: aux_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [sp_index_methodology.md](sp_index_methodology.md)
- [sp_history_simulation.md](sp_history_simulation.md)
- [parse_sp500.md](parse_sp500.md) / [backfill_constituents.md](backfill_constituents.md)
- [reconcile_sp500.md](reconcile_sp500.md)
