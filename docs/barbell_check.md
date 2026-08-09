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
\text{barbell\_score} = \frac{\text{weight}_{\text{convex}} - \text{weight}_{\text{middle}}}{\text{weight}_{\text{safe}} + \text{weight}_{\text{convex}}}
$$

- Positive → barbell (safe + convex, little middle)
- Negative → "Christmas tree" (weight concentrated in fragile middle)

**Fragility-scaled convexity allocation:**

$$
\text{convex\_alloc} = \text{base\_convex} \times (1 + \alpha \cdot \bar{F})
$$

where $\bar{F}$ = average fragility score of current holdings; $\alpha = 0.5$.

**Convex bucket composition:**

| Bucket | Source | Cost metric |
|---|---|---|
| Safe | `fragility_screen` low-fragility names + cash | zero cost |
| Middle | excluded | — |
| Convex | gap risk (gap_risk.py), tail options (options_skew.csv), long volatility (vol_target) | put ladder cost from `options_skew.csv` |

**Put ladder annual cost:**

$$
\text{put\_ladder\_cost\_ann} = \sum_{k} \text{put\_cost}_k \times \frac{252}{\text{days\_to\_expiry}_k}
$$

from `options_skew.csv` (ATM IV, skew, put/call ratios).

**Vol-of-vol beta:**

$$
\text{vol\_of\_vol\_beta} = \frac{\text{cov}(r, \text{vol}^2)}{\text{var}(\text{vol}^2)}
$$

measures convexity payoff in vol-of-vol space.

## Outputs

`barbell_check.csv` — `n_names, weight_safe, weight_middle, weight_convex,
barbell_score, vol_beta, vol_of_vol_beta, avg_atm_iv, put_ladder_cost_ann,
recommended_convexity_alloc`

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