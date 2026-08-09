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
2. **All GICS sub-industries** — every sub-industry with ≥ 2 members having
   price history, id `sub_<subindustry>`.
3. **All factor groups** — every group in `factor_groups.csv` via
   `factor_group_members.csv`, open-ended membership only (valid_to
   null/NaN = current), id `fg_<group>`, kind `factor_group:<group_type>`.

Any ticker already in `daily_prices.parquet` is picked up automatically
when it appears in GICS or factor-group membership — including the former
"amplifiers" (TSM, ASML, SCCO, AEM, BTI, …). No per-basket ticker list
exists in code.

Commodity legs attach by **name pattern** (`COMMODITY_MAP` regexes against
the basket id), not by ticker list — e.g. any basket matching `copper` gets
PCOPPUSDM.

## Outputs

- `macro_sector_shock.csv` — monthly long: `basket, basket_kind, label,
  date, n_members, basket_mom_12m, commodity_mom_12m, shock_score,
  shock_zone`
- `basket_members.csv` — point-in-time membership: `basket, basket_kind,
  label, ticker, commodity`

Zones: basket 12m mom ≥ 0.80 = `shock`, ≥ 0.40 = `elevated`. Score =
z(basket mom) [+ z(commodity mom)].

## Usage

```bash
python macro_sector_shock.py --save
```

Wired into `run_daily_automation.py` as `taleb_sector_shock`; feeds export.

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs
