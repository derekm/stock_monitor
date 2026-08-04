# factor_rotation_defense.py

Defensive factor-rotation strategies across quality / value / low-vol /
dividend / dual-pass sleeves, rotated by a risk-on/off signal.

## Why it exists (rationale)

Rather than a static book, rotate factor sleeves with the regime: overweight
quality/dual in risk-on, low-vol + dividend ETFs in risk-off (high vol or crisis
flag). It is a defensive overlay on top of the screens — a way to de-risk
without leaving the strategy.

## Usage

```bash
python factor_rotation_defense.py --save
```

Flags: `--save`. Reads `daily_prices.parquet`, `monitored_stocks.parquet`,
`fundamentals.parquet`, `preferred_metrics.csv`.

## Outputs

- `factor_rotation_weights.csv` — target sleeve weights
- `factor_rotation_performance.csv` — backtest performance per sleeve
- `factor_sleeve_returns.csv` — sleeve return series

(Schema families: weights_performance / base_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [factor_panel.md](factor_panel.md) — factor scores
- [hmm_regime_detection.md](hmm_regime_detection.md) / [crisis_correlation.md](crisis_correlation.md) — risk-on/off signal
- [preferred_metrics.md](preferred_metrics.md)
