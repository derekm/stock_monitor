# subindustry_regime.py — per-basket crisis regimes (DYNAMIC baskets)

## What it does

Runs the market HMM recipe (vol21 + intra-basket avg pairwise correlation →
3-state stress posterior) on **every dynamic basket** — GICS sectors,
sub-industries, and factor groups. Not a fixed research list.

## Outputs

- `subindustry_regime.csv` — `basket, basket_kind, label, n_members, date,
  vol21, avg_corr, p_stress, regime, ride_pos`
- `subindustry_regime_lead.csv` — `basket, basket_kind, label, n_members,
  n_exits, lead_10d, lead_20d, lead_30d`

## Honest measured answer on "leading collapse"

p_stress > 0.8 fires BEFORE the ride rule's momentum exit only ~14-17% of
the time (10/20/30d leads) — the stress flip is roughly coincident with the
momentum rollover, not leading it. Use the regimes as a CURRENT-STATE map
(which baskets are in high_vol_stress now) to gate NEW ride entries, not to
time exits.

## Usage

```bash
python subindustry_regime.py --save
```

Wired into `run_daily_automation.py` as `taleb_subindustry_regime`
(900s cap — one HMM per basket); feeds export.

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs
