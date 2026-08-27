# risk_enrich.py

Enrich `preferred_metrics` and fundamentals analytics with realized vol, beta,
and max drawdown.

## Why it exists (rationale)

The quality/value screen lacks risk context. This adds realized volatility,
beta (vs the market), and max drawdown to `preferred_metrics.csv` (and a
`risk_metrics.csv`), so sizing and the buy decision can see risk alongside
quality — feeding `vol_target` and `risk_parity_analytics`.

## Usage

```bash
python risk_enrich.py --save
```

Flags: `--save` (writes the enriched `preferred_metrics.csv` + `risk_metrics.csv`).
Reads `daily_prices/`, `monitored_stocks.parquet`, `preferred_metrics.csv`.

## Outputs

- `preferred_metrics.csv` — enriched in place (vol/beta/DD columns)
- `risk_metrics.csv` — standalone risk metrics table

(Schema families: screen_decision / summary_metrics — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [preferred_metrics.md](preferred_metrics.md) — the table it enriches
- [vol_target.md](vol_target.md) / [risk_parity_analytics.md](risk_parity_analytics.md)
- [risk_metrics_ext.md](risk_metrics_ext.md)
