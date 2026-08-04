# regime_aware_constraints.py

Joins HMM regimes with dual-pass policy:

1. **Regime-specific binding** — risk of base dual / near-miss baskets inside each regime  
2. **Transition triggers** — vol/corr/return moves around regime switches  
3. **Regime-aware thresholds** — tighter value in stress, fair-price quality in calm  

```bash
python regime_aware_constraints.py --save
```

Outputs: `regime_constraint_binding.csv`, `hmm_transition_triggers.csv`,
`regime_aware_dual_pass.csv`, `regime_aware_thresholds.json`, `regime_aware_summary.csv`

## Related programs

- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/stress_dual_pass.md](stress_dual_pass.md)
- [docs/hmm_regime_detection.md](hmm_regime_detection.md)
- [docs/portfolio_optimization.md](portfolio_optimization.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
