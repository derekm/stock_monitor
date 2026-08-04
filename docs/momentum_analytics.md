# momentum_analytics.py

Cross-sectional and time-series momentum metrics per ticker, plus quintile
spreads and IC vs forward return.

## Why it exists (rationale)

Momentum is one of the factor inputs to `factor_panel` / `preferred_metrics` and
the buy decision. This computes trailing-horizon returns (21/63/126/252d), a
time-series momentum score (z-scored horizons averaged), the classic 12-1
skip-month momentum, and residual momentum vs the market — then ranks
cross-sectionally and tests IC.

## Usage

```bash
python momentum_analytics.py --universe all --save
python momentum_analytics.py --universe portfolio
```

Flags: `--universe` (index list or `all`, default `all`), `--save`. Reads
`daily_prices.parquet`.

## Outputs

- `momentum_metrics.csv` — per-ticker momentum metrics
- `momentum_quintiles.csv` — cross-sectional quintile spreads
- `momentum_ic.csv` — IC vs forward 21d return

(Schema family: screen_decision — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [factor_panel.md](factor_panel.md) — combines this
- [preferred_metrics.md](preferred_metrics.md)
- [buy_candidates.md](buy_candidates.md)
