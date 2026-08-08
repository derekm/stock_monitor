# barbell_check.py — Barbell structure / convexity

## Description
Checks whether the portfolio is a Taleb barbell (most weight safe, a sliver
convex) or a "Christmas tree" (all middle-risk, exposed to every tail,
rewarded for none). Computes vol beta, vol-of-vol beta, safe/middle/convex
buckets, barbell score, hedge cost, and a recommended convexity allocation.

## Why it exists (rationale)
A portfolio of mid-risk mid-return assets is fragile: it harvests premium in
calm and bleeds in spikes with no offsetting convexity. The barbell (≈90%
ultra-safe + ≈10% highly convex) is the antifragile structure. Negative
vol-of-vol beta = short-vol = fragile; positive = benefits from disorder.

## Usage
```
python barbell_check.py
```

## Outputs (see SCHEMAS → `taleb` family)
- `barbell_check.csv` — n_names, weight_safe/middle/convex, barbell_score
  (positive = barbell, negative = Christmas tree), vol_beta, vol_of_vol_beta,
  avg_atm_iv, put_ladder_cost_ann, recommended_convexity_alloc.

## Related
fragility_screen.py (average fragility scales the allocation),
gap_risk.py (gap share defines the convex bucket), options_skew.csv (hedge
cost). Registered as the `taleb_barbell` daily job (after fragility + ergodic).
