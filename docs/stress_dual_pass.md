# stress_dual_pass.py

Stress-tests the six-leg dual-pass gate (one-at-a-time, scenarios, leave-one-leg-out).

Base universe (5 names): BEN, IVZ, RF, FITB, HBAN. Scenarios swept: `tight`, `base`,
`relaxed_quality`, `relaxed_value`, `relaxed_both`, `buffett_fair` (each relaxes the six
legs differently — e.g. `tight` raises thresholds, `buffett_fair` widens EV/EBITDA≤15,
P/B≤3.0, MktCap/Assets≤1.5). The number of passing names is **data-dependent** (it
recomputes against the current fundamentals snapshot), so do not treat printed counts as
fixed.

```bash
python stress_dual_pass.py --save
```

Output: `dual_pass_stress.csv`

## Related programs

- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/regime_aware_constraints.md](regime_aware_constraints.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
