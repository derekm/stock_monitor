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

- **SMCI ≤ 5%** of total portfolio (vol-target), even if quality looks strong  
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
