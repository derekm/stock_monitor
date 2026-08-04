# crisis_correlation.py

Correlation breakdown in stress / crisis regimes — compares average pairwise
correlation in calm vs crisis windows.

## Why it exists (rationale)

Diversification fails exactly when you need it: in crises, correlations spike.
This script quantifies that by defining crisis windows (top-quintile market
vol, worst 5% return days, drawdown episodes below -8% from peak) and measuring
how pairwise correlation shifts from calm to crisis — the evidence behind the
"hedges and cash buffers matter" thesis.

## Usage

```bash
python crisis_correlation.py --save
```

Flags: `--save` (write outputs). Reads `daily_prices.parquet`,
`monitored_stocks.parquet`.

## Outputs

- `crisis_correlation_summary.csv` — avg pairwise corr: calm vs crisis
- `crisis_correlation_pairs.csv` — per-pair corr change
- `crisis_avg_corr_timeseries.csv` — rolling average correlation over time

(Schema family: correlation_matrix — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [allpairs_correlations.md](allpairs_correlations.md) — full pairwise history
- [cross_asset_analysis.md](cross_asset_analysis.md)
- [hmm_regime_detection.md](hmm_regime_detection.md) — alternative regime definition
- [maintain_analytics.md](maintain_analytics.md)
