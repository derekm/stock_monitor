# rolling_window_analysis.py

Rolling vol, beta, Sharpe, max-drawdown, and dual-screen stability.

## Why it exists (rationale)

Single-point risk stats hide regime dependence. This rolls vol/beta/Sharpe/DD
(and dual-screen pass stability) over a trailing window per ticker, so you can
see whether a name's risk profile is stable or deteriorating — input to sizing
and the buy decision.

## Usage

```bash
python rolling_window_analysis.py --save
```

Flags: `--save`. Reads `daily_prices.parquet`, `monitored_stocks.parquet`,
`portfolio_holdings.parquet`, `preferred_metrics.csv`.

## Outputs

- `rolling_window_metrics.csv` — per-ticker rolling risk metrics

(Schema family: summary_metrics — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [risk_enrich.md](risk_enrich.md) / [risk_metrics_ext.md](risk_metrics_ext.md)
- [preferred_metrics.md](preferred_metrics.md)
- [momentum_analytics.md](momentum_analytics.md)
