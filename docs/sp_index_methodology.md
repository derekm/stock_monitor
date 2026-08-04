# sp_index_methodology.py

S&P 500 inclusion/exclusion reimplementation + our dual-pass strength tiers,
tracked against S&P actuals.

## Why it exists (rationale)

This is the analytics half of the stockmagic ↔ stock_monitor integration: it
reimplements S&P's inclusion/exclusion logic and overlays our own dual-pass
strength tiers, then scores how our tiers line up with real S&P additions/
removals. It turns the index-tracking work from "describe the index" into "score
names the way the index committee would," measured against history.

## Usage

```bash
python sp_index_methodology.py --save
```

Flags: `--save`. Reads `sp500_changes.parquet`, `sp500_constituents.parquet`,
`fundamentals.parquet`, `daily_prices.parquet`, `monitored_stocks.parquet`.

## Outputs

- `sp_index_methodology.csv` — per-name tier vs S&P actual inclusion state
  (and reconciliation stats)

(Schema family: screen_decision — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [sp_universe_tracking.md](sp_universe_tracking.md)
- [sp_history_simulation.md](sp_history_simulation.md) — historical hit/miss
- [parse_sp500.md](parse_sp500.md) / [parse_sp500_changes.md](parse_sp500_changes.md)
- [inclusion_criteria.md](inclusion_criteria.md) — our dual-pass gate
