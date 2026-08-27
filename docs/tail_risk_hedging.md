# tail_risk_hedging.py

Explore tail-risk hedging overlays for the defensive book.

## Why it exists (rationale)

The defensive sleeve still draws down in crises. This evaluates hedging overlays
on the equal-weight defensive index — e.g. a cash buffer (hold 10–20% cash),
put/spread overlays, and rotation to low-vol — and reports the drawdown/return
trade-off so a hedge can be chosen on evidence, not fear.

## Usage

```bash
python tail_risk_hedging.py --save
```

Flags: `--save`. Reads `daily_prices/`, `monitored_stocks.parquet`,
`fundamentals.parquet`.

## Outputs

- `tail_risk_hedge_performance.csv` — per-hedge performance (drawdown, return)
- (and related `tail_risk_hedge_crisis.csv` / summary tables written alongside)

(Schema family: weights_performance — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [build_defensive_index.md](build_defensive_index.md)
- [monte_carlo.md](monte_carlo.md) — terminal-wealth tails
- [regime_aware_constraints.md](regime_aware_constraints.md)
- [crisis_correlation.md](crisis_correlation.md)
