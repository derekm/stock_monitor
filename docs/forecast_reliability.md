# forecast_reliability.py

Rank Granite forecast setups on the actual holdings (from first trade) so you can
pick more reliable configurations.

## Why it exists (rationale)

A single forecast config may look good by chance. This script runs several
`forecast_granite.py backtest` configurations with `--from-first-trade` semantics
over the real holdings and ranks them by reliability — turning "which horizon /
window works" into a comparison table instead of guesswork.

## Usage

```bash
python forecast_reliability.py --index portfolio --save
python forecast_reliability.py --ticker MOS,PFE --horizons 5,10,20 --windows 40,60 --save
```

Flags: `--index` (default portfolio), `--ticker`, `--horizons` (default
`5,10,20`), `--windows` (default `40,60`), `--save`.

## Outputs

- `forecast_reliability_rank.csv` — ranked setups (reads
  `forecast_backtest_metrics.csv`)

(Schema family: forecast_anomaly — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [forecast_granite.md](forecast_granite.md) — the backtests it orchestrates
- [granite_backfill.md](granite_backfill.md) / [ttm_features.md](ttm_features.md)
