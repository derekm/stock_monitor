# binding_constraints_analysis.py

Quantifies the impact of the dual-pass binding constraints — which legs bind,
how many names each leg excludes, and the risk of the resulting baskets.

## Why it exists (rationale)

`inclusion_criteria.py` applies a six-leg gate. This script explains *why* the
gate shapes the book the way it does: per leg it counts failures (alone and
jointly), measures the "shadow dual" set if that leg were removed, computes risk
metrics of the base dual vs leave-one-out baskets, and reports distance-to-
threshold for near-miss names on the binding legs (ROIC, MktCap/Assets, …). It
informs tuning of the inclusion thresholds.

## Usage

```bash
python binding_constraints_analysis.py --save
```

Flags: `--save` (write the CSVs; without it, prints only).

## Outputs

- `binding_constraints_impact.csv` — per-leg failure counts (alone/joint) and
  shadow-dual sizes
- `binding_near_miss_detail.csv` — near-miss names with distance-to-threshold on
  binding legs
- `binding_basket_risk.csv` — risk metrics of base dual vs leave-one-out baskets

(Schema families: screen_decision / summary_metrics — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [inclusion_criteria.md](inclusion_criteria.md) — the six-leg gate
- [stress_dual_pass.md](stress_dual_pass.md) — scenario stress of the same gate
- [regime_aware_constraints.md](regime_aware_constraints.md)
