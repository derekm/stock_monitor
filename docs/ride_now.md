# ride_now.py — CURRENT ride-rule state + recommendation per dynamic basket + ticker

## What it does

For every dynamic basket (GICS sector / sub-industry / factor group) **and
every ticker** in the price universe, computes the ride rule's CURRENT
state and emits an honest recommendation with plain-English interpretation.

## Formulas

Same ride rule as [shock_ride.md](shock_ride.md):

**Monthly returns & momenta (current month):**

$$
mom_{12}(t) = \frac{C(t)}{C(t-12)} - 1
\quad
mom_3(t) = \frac{C(t)}{C(t-3)} - 1
\quad
mom_1(t) = \frac{C(t)}{C(t-1)} - 1
$$

**Current ride position (same lagged logic as shock_ride):**

$$
ride_long = \mathbb{1}\left[mom_{12} > 0.40 \;\land\; mom_3 > 0\right]
$$

**Honest recommendation (plain-English interpretation surfaced in dashboard):**

| Recommendation | Condition | Interpretation |
|---|---|---|
| **BUY** | ride_long ∧ mom12 > 0.40 ∧ mom3 > 0 ∧ mom1 > 0 | "explosion still accelerating (12m X%, 3m Y%, 1m Z%); regime calm — cleanest ride-long now" |
| **STAND DOWN** | ride_long ∧ (mom3 ≤ 0 ∨ mom1 ≤ 0) | "momentum says long (12m X%, 3m Y%, 1m Z%) but 1m rolling over — tighten stop to 3m rollover" |
| **AVOID** | ¬ride_long ∧ mom12 > 0.40 | "exploded (12m X%) but 3m already Y% (1m Z%) — rolled over; ride exited" |
| **WATCH** | mom12 > 0.40 ∧ mom3 > 0 ∧ ¬ride_long | "above threshold but 3m not yet positive; waiting for entry" |
| **FLAT** | otherwise | "no signal" |

The tension is surfaced: **STAND DOWN** means the momentum rule says long
but the basket's own HMM stress regime is maxed (p_stress ≈ 1.0). The
contradiction *is* the signal.

## Outputs

`ride_now.csv` — per basket/ticker: `basket|ticker, basket_kind, label,
n_members, date, mom1, mom3, mom12, ride_long, shock_zone, p_stress,
regime, recommendation, interpretation`

## Usage

```bash
python ride_now.py --save
```

Wired into `run_daily_automation.py` as `taleb_ride_now`; feeds export.

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [shock_ride.md](shock_ride.md) — ride rule + historical stats
- [subindustry_regime.md](subindustry_regime.md) — p_stress source
- [macro_sector_shock.md](macro_sector_shock.md) — dynamic baskets + shock zones
- [hmm_regime_detection.md](hmm_regime_detection.md) — p_stress source
- [export_dashboard_data.md](export_dashboard_data.md) — catalog + dashboard wiring