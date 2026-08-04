# inclusion_criteria.py

Automated **inclusion / exclusion** policy for the portfolio system.

## Dual-pass (INCLUDE_CORE)

Must clear **all** six legs:

| Leg | Threshold |
|-----|-----------|
| ROE | ≥ 15% |
| ROIC | ≥ 15% |
| Debt/Equity | ≤ 1.0 |
| EV/EBITDA | ≤ 9 |
| P/B | ≤ 1.5 |
| MktCap/Assets | ≤ 0.5 |

## Other bands

| Decision | Rule |
|----------|------|
| INCLUDE_VALUE | Trifecta only |
| INCLUDE_QUALITY | Buffett only |
| NEAR_DUAL | Fail 1–2 legs |
| WATCH / AVOID | Weaker composites |

## Hard policy overlays

- Per-name weight caps (vol-target); even strong-quality names stay within the cap  
- Very low `earnings_stability` blocks CORE promotion  
- Growth-tech names are not auto-promoted to CORE without dual pass  

## Outputs

- `inclusion_candidates.csv` / `exclusion_candidates.csv` / `near_dual_candidates.csv`
- `defensive_value_exploration.csv`
- `inclusion_rules.json`
- `asset_correlation_matrix.csv` / `sector_correlation_matrix_latest.csv`

```bash
python inclusion_criteria.py --explore-defensive --save
```

Rules are loaded into the dashboard **Inclusion Rules** tab.

## Related programs

- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/stress_dual_pass.md](stress_dual_pass.md)
- [docs/regime_aware_constraints.md](regime_aware_constraints.md)
- [docs/dual_screen_analysis.md](dual_screen_analysis.md)
- [docs/build_defensive_index.md](build_defensive_index.md)
- [docs/manage_stocks.md](manage_stocks.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
