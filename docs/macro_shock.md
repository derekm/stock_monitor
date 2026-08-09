# macro_shock.py

Macro supply-shock layer — the oil-crisis complement to `macro_fragility.py`.

## Why it exists (rationale)

`macro_fragility.py` measures the DEMAND side of crises (debt impulse,
Credit Accelerator, Minsky signal). It catches debt-driven crises (1987,
2000, 2008, 2020, 2022) but structurally MISSED the 1973-74 oil crisis —
the one miss in the crisis-label validation (impulse 0.162, danger not
crisis_band). That miss is not a bug: 1973-74 was a SUPPLY shock, and a
debt-driven signal cannot see one. This layer is the supply-side twin.

## Analytics (all verified to have fired in 1973-74)

1. **Oil momentum** — 12m change in crude price (spliced IMF OILPRICE
   1946-2013 + WTI 1986-). The embargo quadrupled oil: +184% YoY by
   Mar-1974. Fired at every oil crisis: 1973-74, 1979-80 (+119%),
   2008 (oil → $140), 2022 invasion.
2. **Inflation surprise** — CPI YoY vs trailing 3y norm. Supply shocks
   show as inflation above the recent norm (1974: ~12% YoY vs ~4% norm);
   distinguishes supply from demand shocks.
3. **Real rate** — fed funds minus CPI YoY. Deeply negative real rates are
   the supply-shock signature (1974-75, 2022 at −7.8%).
4. **Energy divergence** — 12m energy-producer basket vs equal-weight
   market. Energy rising while the market falls = supply shock. Caveat:
   only 2-3 long-history names in 1973 (XOM/CVX/HAL) make this leg noisy
   in the early era (adding HAL flips the sign in 1973); it strengthens
   after 1980 (10 names). Reported separately, NOT in the composite score.

## Composite and zones

Shock score = z(oil_mom) + z(inflation_surprise) − z(real_rate), the three
robust legs. Zones calibrated on OUR crisis history:

- `shock` — oil_mom ≥ +40% AND/OR score ≥ 1.5σ (1973-74: 2.74, 1979-80:
  2.04, 2008: 1.61, 2022: 2.50)
- `elevated` — oil_mom ≥ +15% or score ≥ 0.75σ
- `benign` — otherwise. Oil-price COLLAPSES read benign on purpose (1986
  −55%, 2014-15 −50%, 2020 −74% are deflationary events, correctly caught
  by the demand layer, not this one).

## Outputs

`macro_shock.csv` — monthly (60y window):
`date, oil_mom_12m, inflation_surprise, real_rate, energy_divergence,
 shock_score, shock_zone`

Reads: FRED CSV (network, cached under `macro_data/` shared with
`macro_fragility.py`), `daily_prices.parquet` (energy basket).

## Usage

```bash
python macro_shock.py --save
```

Wired into `run_daily_automation.py` as the `taleb_shock` job (depends on
`hmm`; feeds `export`).

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs
