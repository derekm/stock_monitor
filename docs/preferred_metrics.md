# preferred_metrics.py

Automated **preferred metrics** for screening, sizing, and inclusion decisions.

## Metric stack

### Buffett-style quality
| Metric | Prefer |
|--------|--------|
| ROE | ≥ **15%** |
| ROIC | ≥ **15%** |
| Debt/Equity | ≤ **1.0** (ideal ≤ 0.5) |
| Interest coverage | higher better |
| Earnings stability | 0–1 score (predictability) |

### Value trifecta (prior threads)
| Metric | Prefer |
|--------|--------|
| EV/EBITDA | ≤ **9** |
| P/B | ≤ **1.5** |
| MktCap/Assets | ≤ **0.5** |

### Sizing overlays
- Composite score → suggested max weight bands (3–12%)
- Per-name weight caps (vol-target aware when `vol_targets.csv` present)
- Actions: `prefer_add` / `hold_or_add` / `hold` / `reduce_or_avoid`

### Cash-distrust discount
- `distrust_p_bad` — heuristic P(bad outcome) from decline stage, ARISTA flag, and quality.
- `distrust_discount` = `1 − distrust_p_bad × excess_cash_share`. `buy_candidates.py` multiplies its own score by this, clipped to [0.5, 1.0], so it moves live BUY/ACCUMULATE/WATCH labels. Note `composite_score` here is computed *before* the discount and is not scaled by it.
- `distrust_p_bad_fitted` — logit fit, **diagnostic only**. It failed honest walk-forward validation (`distrust_oos_eval.py`): pooled OOS AUC **0.591** on a ≥$5M/day liquid universe, under the 0.65 gate and beaten by trailing volatility alone (`vol63` 0.651). It is deliberately NOT blended into `distrust_p_bad`; `test_basic.py::test_distrust_fit` fails if a re-blend is reintroduced.
- `distrust_fit_auc_insample` — the old in-script number (~0.65). NOT out-of-sample: it splits rows by alphabetical ticker while every label comes from the same final 63-day window. Kept for continuity; do not gate on it. `distrust_fit_auc_oos` (0.591) and `distrust_fit_gate_pass` (False) record the honest result.

## Decision labels

| Label | Meaning |
|-------|---------|
| **INCLUDE_CORE** | Buffett pass **and** trifecta pass |
| **INCLUDE_VALUE** | Trifecta pass |
| **INCLUDE_QUALITY** | Buffett pass |
| **SATELLITE** | Solid composite, not both screens |
| **WATCH** / **AVOID** | Weak composite |

The `INCLUDE_CORE` set is also read by [rebalance_calendar.md](rebalance_calendar.md)
as the dual-core count (`n_dual_core` column) and by
[regime_aware_constraints.md](regime_aware_constraints.md) as the base dual-pass basket.

```bash
python preferred_metrics.py --seed-quality --save
python preferred_metrics.py --decision INCLUDE_VALUE
python check_alerts.py --dry-run   # VALUE_TRIFECTA + BUFFETT_QUALITY rules
```

Outputs: `preferred_metrics.csv`, `preferred_screen_hits.csv`  
Quality fields are seeded as **approx** until live fundamentals replace them (`quality_source=seed_approx_buffett`).

## Related programs

- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/update_fundamentals.md](update_fundamentals.md)
- [docs/fundamentals_history.md](fundamentals_history.md)
- [docs/dupont_analysis.md](dupont_analysis.md)
- [docs/vol_target.md](vol_target.md)
- [docs/kelly.md](kelly.md)
- [docs/risk_metrics_ext.md](risk_metrics_ext.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
