# regime_aware_constraints.py

Regime-specific constraint binding — how the dual / near-miss baskets behave
inside each HMM regime.

## Why it exists (rationale)

Caps and gates that look fine in calm markets can blow up in stress. This shows,
per HMM regime, how the dual-pass and near-miss baskets behave: vol, drawdown,
and hit-rate of each leg treated as a risk filter — the evidence for tightening
constraints in high_vol_stress. Pairs with `regime_correlation_breakdown`.

## Usage

```bash
python regime_aware_constraints.py --save
```

Flags: `--save`. Reads `daily_prices.parquet`, `fundamentals.parquet`,
`monitored_stocks.parquet`, `hmm_regime_states.csv`, `preferred_metrics.csv`.

## Outputs

- `regime_constraint_binding.csv` — per-regime binding metrics (and related tables
  written around line 277)

(Schema family: regime_state — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [regime_correlation_breakdown.md](regime_correlation_breakdown.md)
- [hmm_regime_detection.md](hmm_regime_detection.md) / [hmm_posterior_analysis.md](hmm_posterior_analysis.md)
- [inclusion_criteria.md](inclusion_criteria.md) / [binding_constraints_analysis.md](binding_constraints_analysis.md)
- [rebalance_calendar.md](rebalance_calendar.md) — calendar reads the same `hmm_regime_states.csv`
  regime label directly (it does **not** consume `rebalance_calendar.csv`; that file is
  currently an orphan output — see its correctness notes)
