# subindustry_regime.py

Per-sub-industry correlation & crisis regimes, plus an honest answer to
"how do we find collapsing sectors before the ride rule takes losses?"

## What it computes

For each focused subsector basket (the `sub_*` entries of
macro_sector_shock.SECTORS — fertilizers, E&P, copper, gold, steel, semis,
semi-equip, biotech, etc.):

- **basket returns** (daily, equal-weight, best-available history)
- **vol21** — 21d rolling vol of the basket (annualized)
- **avg_corr** — 21d rolling average pairwise correlation WITHIN the basket
  (the "correlation regime" per sub-industry)
- **3-state HMM** (same code as hmm_regime_detection — build_features /
  fit_hmm / label_states, no fork) labeled low_vol / normal /
  high_vol_stress. p_stress = posterior probability of the stress state
  (the "crisis regime" per sub-industry).
- **ride_pos** — the shock_ride rule's position at each date, so the two
  can be compared point-in-time.

## The measured answer to "find collapsing sectors before the ride rule"

Two candidate leading signals tested against every ride-rule EXIT
(position 1→0) across all 24 subsectors, full history:

1. **Subsector HMM stress flip**: p_stress > 0.8 fires BEFORE the ride
   exit only 14-17% of the time (10/20/30d leads). The stress flip is
   roughly COINCIDENT with the momentum rollover — it does not lead it.
2. **Intra-basket correlation regime**: at ride exits, correlation sits at
   the 58th percentile of its trailing 2y — mildly elevated, not a
   leading edge.

Conclusion (honest): the shock_ride momentum exit (3m rollover) is already
near-optimal in timing; no stress/correlation variant beats its drawdown
(a stress-as-exit variant is simply buy-and-hold: long 75-97% of the time,
maxDD −56% vs ride's −42%). The per-subsector regimes are NOT a
prediction layer — they are a CURRENT-STATE map: which subsectors are in
high_vol_stress right now (live: semi_equip 1.00, med_devices 1.00, semis
0.98, copper 0.59). Use them to GATE NEW ride entries away from
already-stressed subsectors, not to time exits.

## Outputs

- `subindustry_regime.csv` — basket, date, vol21, avg_corr, p_stress,
  regime, ride_pos (daily, per subsector)
- `subindustry_regime_lead.csv` — basket, n_exits, lead_10d/20d/30d
  (count of exits where p_stress was already > 0.8 that many days before)

Reads: daily_prices.parquet, sp500_constituents.parquet (via
macro_sector_shock), hmm_regime_detection (imported, same HMM code).

## Usage

```bash
python subindustry_regime.py --save
```

Wired into `run_daily_automation.py` as the `taleb_subindustry_regime` job
(depends on `taleb_sector_shock`; feeds `export`).

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs
