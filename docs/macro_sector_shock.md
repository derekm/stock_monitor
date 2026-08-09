# macro_sector_shock.py

Sector shock signals — farming inputs/outputs, materials, and any
basket-vs-commodity sector, using the `macro_shock.py` recipe generalized.

## Why it exists (rationale)

`macro_shock.py` proved the supply-shock framework on oil (1973-74,
1979-80, 2008, 2022 all fired). The user asked whether fertilizer/materials
and other economic sectors show the same shock signals. Verified in OUR
data before building:

- **Farming inputs** (fertilizer basket CF/MOS/NTR/UAN/IPI/LXU/CTVA):
  12m basket momentum peaked **+232% (Oct-2007)** — the run-up to the 2008
  food crisis — and +213% (Nov-2021, the fertilizer supercycle). Same
  explosion signature as the oil crisis.
- **Farming outputs** (grain merchants ADM/BG/SYY): +86% (Nov-1975, the
  1970s food-crisis era), +79% (1983, the farm-debt-crisis era).
- **Materials** (SECT_MATERIALS + IMF all-commodities index): +106%
  (May-2021, commodities supercycle).

## Design

Table-driven: each sector = (equity basket from daily_prices or
sector_prices) + (optional IMF global commodity price). Sector shock score
= z(basket 12m momentum) + z(commodity 12m momentum) — the macro_shock
recipe, minus the inflation/real-rate legs (those are macro-wide and stay
in macro_shock.py). Zones calibrated on the verified events: basket 12m
momentum ≥ +80% = `shock` (fertilizer 2007/2021 both exceeded +200%),
≥ +40% = `elevated`.

Sectors (add more by appending to the SECTORS table — no code change):

| Sector | Basket | Commodity |
|---|---|---|
| `farming_inputs` | CF/MOS/NTR/UAN/IPI/LXU/CTVA | none on FRED (basket carries it) |
| `farming_outputs` | ADM/BG/SYY | PWHEAMTUSDM (global wheat) |
| `materials` | SECT_MATERIALS (sector_prices) | PALLFNFINDEXM (all commodities) |

## Outputs

`macro_sector_shock.csv` — monthly (60y window):
`sector, date, basket_mom_12m, commodity_mom_12m, shock_score, shock_zone`

Reads: daily_prices.parquet / sector_prices.parquet, FRED CSV (cached
under macro_data/, shared with the other macro scripts).

## Usage

```bash
python macro_sector_shock.py --save
```

Wired into `run_daily_automation.py` as the `taleb_sector_shock` job
(depends on `hmm`; feeds `export`).

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs
