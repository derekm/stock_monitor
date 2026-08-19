# subindustry_regime.py — per-basket crisis regimes (DYNAMIC baskets)

## What it does

Runs the market HMM recipe (vol21 + intra-basket avg pairwise correlation →
3-state stress posterior) on **every dynamic basket** — GICS sectors,
sub-industries, and factor groups. Not a fixed research list.

## Formulas

**Daily equal-weight basket returns:**

$$
r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)
\qquad
\bar{r}_t = \frac{1}{|B_t|} \sum_{i \in B_t} r_{i,t}
$$

**21-day rolling annualized volatility:**

$$
vol_{21}(t) = \sqrt{252} \cdot std\left(\bar{r}_{t-20:t}\right)
$$

**21-day rolling average pairwise correlation (intra-basket):**

For each day $t$, compute the correlation matrix of member returns over the
trailing 21 days, then average the upper triangle:

$$
avg_corr(t) = \frac{2}{k(k-1)} \sum_{i<j} corr\left(r_{i,t-20:t}, r_{j,t-20:t}\right)
$$

**HMM features (daily):**

$$
X_t = \left[mkt_ret_t,\; vol_{21}(t),\; avg_corr(t)\right]
$$

Fit a 3-state Gaussian HMM on $X_t$ (full covariance, 200 iterations).
Label states by their mean (vol, corr): `low_vol`, `normal`, `high_vol_stress`.

**Stress posterior:**

$$
p(stress \mid \mathcal{F}_t) = P(state = high_vol_stress \mid X_{1:t})
$$

from the HMM forward-backward algorithm.

**Lead test (stress leading ride exit):**

At every ride exit (3m mom rollover), record whether $p(stress) > 0.8$
at $t-10, t-20, t-30$ days before the exit.

## Honest measured answer on "leading collapse"

p_stress > 0.8 fires BEFORE the ride rule's momentum exit only ~14-17% of
the time (10/20/30d leads) — the stress flip is roughly coincident with the
momentum rollover, not leading it. Use the regimes as a CURRENT-STATE map
(which baskets are in high_vol_stress now) to gate NEW ride entries, not to
time exits.

## Outputs

- `subindustry_regime.csv` — `basket, basket_kind, label, n_members, date,
  vol21, avg_corr, p_stress, regime, ride_pos`
- `subindustry_regime_lead.csv` — `basket, basket_kind, label, n_members,
  n_exits, lead_10d, lead_20d, lead_30d`

## Usage

```bash
python subindustry_regime.py --save
```

Wired into `run_daily_automation.py` as `taleb_subindustry_regime`
(900s cap — one HMM per basket); feeds export.

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [hmm_regime_detection.md](hmm_regime_detection.md) — shared HMM machinery
- [shock_ride.md](shock_ride.md) — ride rule + per-basket stress test (lead test)
- [macro_sector_shock.md](macro_sector_shock.md) — dynamic baskets source
- [shock_ride.md](shock_ride.md) — ride rule + per-basket/ticker stress gate
- [ride_now.md](ride_now.md) — current-state recommendations per basket/ticker
- [hmm_regime_detection.md](hmm_regime_detection.md) — shared HMM machinery
- [macro_sector_shock.md](macro_sector_shock.md) — dynamic baskets source