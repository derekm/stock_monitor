# regime_correlation_breakdown.py

Correlation structure inside each HMM regime.

## Why it exists (rationale)

Diversification depends on correlation, and correlation depends on regime. This
computes average/median pairwise asset correlation *within* each regime (and the
calm-vs-stress split), quantifying how much hedging power survives in a crisis.
Companion to `crisis_correlation` (which uses vol/return windows) and
`regime_aware_constraints`.

## Usage

```bash
python regime_correlation_breakdown.py --save
```

Flags: `--save`. Reads `daily_prices/`, `monitored_stocks.parquet`,
`hmm_regime_states.csv`.

## Outputs

- `regime_corr_breakdown.csv` — per-regime correlation summary

(Schema family: correlation_matrix — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [crisis_correlation.md](crisis_correlation.md)
- [regime_aware_constraints.md](regime_aware_constraints.md)
- [hmm_regime_detection.md](hmm_regime_detection.md)
- [allpairs_correlations.md](allpairs_correlations.md)
