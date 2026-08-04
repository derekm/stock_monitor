# rebalance_calendar.py

Regime- and dual-pass-aware rebalance schedule.

## Why it exists (rationale)

Rebalancing shouldn't be blind monthly. This builds a calendar of rebalance
dates (default: last trading day of month from the `daily_prices` calendar) and
can skip/reduce a rebalance when the current regime is high_vol_stress
(optional half-band), so the book isn't churned into a volatility spike.

## Usage

```bash
python rebalance_calendar.py --months 18 --save
```

Flags: `--months` (default 18), `--save`. Reads `daily_prices.parquet`,
`hmm_regime_states.csv`.

## Outputs

- `rebalance_calendar.csv` — scheduled rebalance dates + regime context

(Schema family: aux_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [hmm_regime_detection.md](hmm_regime_detection.md) — regime input
- [regime_aware_constraints.md](regime_aware_constraints.md)
- [portfolio_optimization.md](portfolio_optimization.md) / [vol_target.md](vol_target.md)
