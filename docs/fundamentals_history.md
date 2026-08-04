# fundamentals_history.py

Time-series fundamentals & preferred-metric snapshots for screen backtests.

## Why it exists (rationale)

Inclusion screens need *history*, not only the latest row. `fundamentals.parquet`
already stores dated (`as_of_date`) rows; this script backfills synthetic history
for robust thesis backtests, snapshots `preferred_metrics` scores through time,
and evaluates screen pass/fail on each `as_of_date` so the dual-pass can be
backtested.

## Usage

```bash
python fundamentals_history.py backfill --quarters 8
python fundamentals_history.py snapshot --save
python fundamentals_history.py screen-backtest --save
```

Subcommands: `backfill`, `snapshot`, `screen-backtest`. Flags: `--quarters`
(default 8), `--save`.

## Outputs

- `preferred_metrics_history.parquet` / `.csv` — score snapshots through time
- `screen_backtest.csv` — pass/fail per ticker per `as_of_date`

(Schema families: base_table / screen_decision — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [preferred_metrics.md](preferred_metrics.md) — scores it snapshots
- [inclusion_criteria.md](inclusion_criteria.md) — gate it backtests
- [dual_screen_analysis.md](dual_screen_analysis.md)
- [backfill_constituents.md](backfill_constituents.md) — real PIT fundamentals
