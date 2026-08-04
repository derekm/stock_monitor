# sp_history_simulation.py

Reproduce S&P 500 inclusion/exclusion decisions in our independent simulation,
and track our reimplementation vs the actuals.

## Why it exists (rationale)

The S&P-tracking subsystem isn't just descriptive — it should *predict* index
events. This simulates inclusion/exclusion decisions with our own scored rules
(from `sp_index_methodology`) over history, then compares our calls to the real
ADD/REMOVE events in `sp500_changes.parquet` so we can measure how well the
reimplementation tracks S&P. This is the historical backtest half of the
stockmagic ↔ stock_monitor integration.

## Usage

```bash
python sp_history_simulation.py --save
```

Flags: `--save`. Reads `sp500_changes.parquet`, `sp500_constituents.parquet`,
`fundamentals.parquet`, `daily_prices.parquet`.

## Outputs

- `sp_history_simulation.csv` — our simulated decisions vs S&P actuals
  (per-event hit/miss)

(Schema family: screen_decision — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [sp_index_methodology.md](sp_index_methodology.md) — the rules it simulates
- [sp_universe_tracking.md](sp_universe_tracking.md)
- [parse_sp500_changes.md](parse_sp500_changes.md) — the actuals
- [reconcile_sp500.md](reconcile_sp500.md)
