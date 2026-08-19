# macro_sector_shock.py — DYNAMIC baskets

Sector / subsector / factor-group shock signals. **Baskets are built at run
time — not a fixed research ticker list.**

## Why it exists

The supply-shock framework (macro_shock.py) proved on oil; this extends it
to every sector. After the user asked for **dynamic baskets and all tickers
everywhere**, the fixed `SECTORS` dict was replaced by run-time construction.

## Dynamic basket sources (no hard-coded ticker lists)

1. **All GICS sectors** — every sector in `sp500_constituents.parquet`
   (current members), id `gics_<sector>`.
2. **All GICS sub-industries** — every sub-industry with >= 2 members having
   price history, id `sub_<subindustry>`.
3. **All factor groups** — every group in `factor_groups.csv` via
   `factor_group_members.csv`, open-ended membership only (valid_to
   null/NaN = current), id `fg_<group>`, kind `factor_group:<group_type>`.

Any ticker already in `daily_prices.parquet` is picked up automatically
when it appears in GICS or factor-group membership — including the former
"amplifiers" (TSM, ASML, SCCO, AEM, BTI, ...). No per-basket ticker list
exists in code.

Commodity legs attach by **name pattern** (`COMMODITY_MAP` regexes against
the basket id), not by ticker list — e.g. any basket matching `copper` gets
PCOPPUSDM.

## Formulas

**Monthly basket momentum (12m):**

$$
mom_{12}(t) = \frac{C(t)}{C(t-12)} - 1
$$

where

$$
C(t) = \prod_{\tau=1}^t \left(1 + \bar{r}_\tau\right)
$$

and

$$
\bar{r}_\tau = \frac{1}{|B_\tau|} \sum_{i \in B_\tau} r_{i,\tau}
$$

B_tau = available basket members at month tau; r_i,tau = monthly log return of member i.

**Shock score (z-standardized composite):**

If commodity mapped:
$$
shock_score = \frac{z(mom_{12,basket}) + z(mom_{12,commodity})}{2}
$$
Otherwise:
$$
shock_score = z(mom_{12,basket})
$$

where $z(x) = \frac{x - \mu_x}{\sigma_x}$ over the full history of the basket.

**Shock zones (calibrated on verified explosions — oil 1973/1979/2008/2022, fertilizer 2007/2021):**

| Zone | Condition |
|---|---|
| `shock` | $mom_{12,basket} \geq 0.80$ |
| `elevated` | $0.40 \leq mom_{12,basket} < 0.80$ |
| `benign` | $mom_{12,basket} < 0.40$ |

**Commodity mapping** (`COMMODITY_MAP` regexes against basket id, first match wins):
- `copper|sub_copper|industry_copper` → PCOPPUSDM
- `nickel` → PNICKUSDM
- `zinc|industrial.?metal` → PZINCUSDM
- `wheat|grain|farming_output|agricultural product` → PWHEAMTUSDM
- `corn|maize` → PMAIZMTUSDM
- `soy` → PSOYBUSDM
- `sugar` → PSUGAISAUSDM
- `cotton` → PCOTTINDUSDM
- `cocoa` → PCOCOUSDM
- `coffee` → PCOFFOTMUSDM
- `rubber` → PRUBBUSDM
- `coal|thermal` → PCOALAUUSDM
- `uranium` → PURANUSDM
- `gas|natural.?gas|midstream|oil.?gas storage` → PNGASUSUSDM
- `^gics_energy$|sector_energy|energy_equit` → PNGASUSUSDM
- `^gics_materials$|sector_materials|^materials$` → PALLFNFINDEXM

## Outputs

- `macro_sector_shock.csv` — monthly long: `basket, basket_kind, label,
  date, n_members, basket_mom_12m, commodity_mom_12m, shock_score,
  shock_zone`
- `basket_members.csv` — point-in-time membership: `basket, basket_kind,
  label, ticker, commodity`

## Usage

```bash
python macro_sector_shock.py --save
```

Wired into `run_daily_automation.py` as `taleb_sector_shock`; feeds export.

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [macro_shock.md](macro_shock.md) — supply-shock twin (oil-only)
- [macro_fragility.md](macro_fragility.md) — debt-fragility twin (demand side)
- [shock_ride.md](shock_ride.md) — rides the explosions this layer labels
- [subindustry_regime.md](subindustry_regime.md) — per-basket HMM stress posteriors
- [hmm_regime_detection.md](hmm_regime_detection.md) — stress posterior source
- [export_dashboard_data.md](export_dashboard_data.md) — catalog + dashboard wiring