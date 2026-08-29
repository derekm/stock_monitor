# barbell_check.py

Barbell portfolio construction — allocate to a "safe" bucket + a "convex"
bucket, skip the fragile middle. The average fragility score scales the
convexity allocation.

## Why it exists (rationale)

The Taleb layer says: hold safe assets (low fragility) + convex bets
(gap risk, tail options), skip the fragile middle. The average fragility
of the portfolio scales the convex allocation: high fragility → more
convexity needed.

## Formulas

**Barbell score (portfolio shape):**

$$
barbell_score = \frac{weight_convex - weight_middle}{weight_safe + weight_convex}
$$

- Positive → barbell (safe + convex, little middle)
- Negative → "Christmas tree" (weight concentrated in fragile middle)

**Fragility-scaled convexity allocation:**

$$
convex_alloc = base_convex \times (1 + \alpha \cdot \bar{F})
$$

where $\bar{F}$ = average fragility score of current holdings; $\alpha = 0.5$.

**Convex bucket composition:**

- Safe: `fragility_screen` low-fragility names + cash (zero cost)
- Middle: excluded
- Convex: gap risk (gap_risk.py), tail options (options_skew.parquet), long volatility (vol_target) — put ladder cost from `options_skew.parquet`

**Put ladder annual cost:**

$$
put_ladder_cost_ann = \sum_{k} put_cost_k \times \frac{252}{days_to_expiry_k}
$$

from `options_skew.parquet` (ATM IV, skew, put/call ratios).

**Vol-of-vol beta:**

$$
vol_of_vol_beta = \frac{cov(r, vol^2)}{var(vol^2)}
$$

measures convexity payoff in vol-of-vol space.

## Outputs

`barbell_check.parquet` — columns: n_names, weight_safe, weight_middle, weight_convex,
barbell_score, vol_beta, vol_of_vol_beta, avg_atm_iv, put_ladder_cost_ann,
recommended_convexity_alloc

## Usage

```bash
python barbell_check.py --save
```

Registered as the `taleb_barbell` daily job (after `fragility` + `ergodic`).

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [fragility_screen.md](fragility_screen.md) — average fragility scales the allocation
- [gap_risk.md](gap_risk.md) — gap share defines the convex bucket
- [options_skew.md](options_skew.md) — put ladder cost
- [vol_target.md](vol_target.md) — long volatility as convex bucket