# macro_shock.py

Macro supply-shock layer — the oil-crisis complement to `macro_fragility.py`.

## Why it exists (rationale)

`macro_fragility.py` measures the DEMAND side of crises (debt impulse,
Credit Accelerator, Minsky signal). It catches debt-driven crises (1987,
2000, 2008, 2020, 2022) but structurally MISSED the 1973-74 oil crisis —
the one miss in the crisis-label validation (impulse 0.162, danger not
crisis_band). That miss is not a bug: 1973-74 was a SUPPLY shock, and a
debt-driven signal cannot see one. This layer is the supply-side twin.

## Formulas

**Oil momentum (12m):**

$$
oil_mom_12(t) = \frac{O(t)}{O(t-12)} - 1
$$

where O(t) = crude price (spliced IMF OILPRICE 1946-2013 + WTI 1986-).

**Inflation surprise:**

$$
inflation_surprise(t) = \pi(t) - \frac{1}{36} \sum_{s=1}^{36} \pi(t-s)
$$

where \pi(t) = CPI YoY; the trailing 3-year average is the "norm".

**Real rate (ex-post):**

$$
real_rate(t) = fed_funds(t) - \pi(t)
$$

Deeply negative real rates are the supply-shock signature (1974-75, 2022
at −7.8%).

**Energy divergence (12m energy-producer basket vs equal-weight market):**

$$
energy_divergence(t) = mom_12_energy(t) - mom_12_market(t)
$$

Energy rising while the market falls = supply shock. Caveat: only 2-3
long-history names in 1973 (XOM/CVX/HAL) make this leg noisy in the early
era; it strengthens after 1980 (10 names). Reported separately, NOT in
the composite score.

**Composite shock score (z-standardized):**

$$
shock_score = z(oil_mom_12) + z(inflation_surprise) - z(real_rate)
$$

The three robust legs (oil momentum, inflation surprise, negative real
rate). Energy divergence is NOT in the composite — reported separately.

**Shock zones (calibrated on OUR crisis history):**

| Zone | Condition |
|---|---|
| `shock` | oil_mom_12m ≥ +40% **OR** score ≥ 1.5σ (1973-74: 2.74, 1979-80: 2.04, 2008: 1.61, 2022: 2.50) |
| `elevated` | oil_mom_12m ≥ +15% **OR** score ≥ 0.75σ |
| `benign` | otherwise |

Oil-price COLLAPSES read benign on purpose (1986 −55%, 2014-15 −50%,
2020 −74% are deflationary events, correctly caught by the demand layer,
not this one).

## Data

FRED public CSV endpoints (cached under `macro_data/`, shared with
`macro_fragility.py`):

- Oil: `OILPRICE` (IMF, 1946-2013) + `DCOILWTICO` (WTI, 1986-)
- CPI: `CPIAUCSL` (YoY)
- Fed funds: `FEDFUNDS` (effective federal funds rate)

Cached under `macro_data/` (TTL 35d; FRED publishes with ~1 quarter lag).

## Outputs

`macro_shock.csv` — monthly (60y window):

```
date, oil_mom_12m, inflation_surprise, real_rate, energy_divergence,
shock_score, shock_zone
```

## Usage

```bash
python macro_shock.py --save
```

Reads: FRED CSV (network, cached under `macro_data/` shared with
`macro_fragility.py`), `daily_prices.parquet` (energy basket).

Wired into `run_daily_automation.py` as the `taleb_shock` job (depends on
`hmm`; feeds `export`).

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [macro_fragility.md](macro_fragility.md) — demand-fragility twin (debt side)
- [macro_sector_shock.md](macro_sector_shock.md) — sector/subsector extensions
- [shock_ride.md](shock_ride.md) — rides the explosions this layer labels
- [hmm_regime_detection.md](hmm_regime_detection.md) — stress posterior (shared)
- [export_dashboard_data.md](export_dashboard_data.md) — catalog + dashboard wiring