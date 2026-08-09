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

Table-driven: each sector = (equity basket from daily_prices, loaded from
**full S&P 500 GICS membership** when `gics` is set — sp500_constituents,
so coverage never depends on hand-picked thin lists — plus optional extra
tickers) + (optional IMF global commodity price). Sector shock score
= z(basket 12m momentum) + z(commodity 12m momentum) — the macro_shock
recipe, minus the inflation/real-rate legs (those are macro-wide and stay
in macro_shock.py). Zones calibrated on the verified events: basket 12m
momentum ≥ +80% = `shock` (fertilizer 2007/2021 both exceeded +200%),
≥ +40% = `elevated`.

Basket sourcing (2026-08): GICS sectors (Energy, Materials, Consumer
Staples) load the full S&P membership dynamically. `farming_inputs` uses
its FOCUSED basket (CF/MOS/NTR/UAN/IPI/LXU/CTVA) — measured: the broad
Materials GICS smears the fertilizer explosion signal (peak drops from
+232% focused to +91% broad), so concentrated sub-sectors keep explicit
baskets.

Sectors (add more by appending to the SECTORS table — no code change):

| Sector | Basket | Commodity |
|---|---|---|
| `farming_inputs` | focused fertilizer names | none on FRED (basket carries it) |
| `farming_outputs` | Consumer Staples GICS | PWHEAMTUSDM (wheat) |
| `materials` / `copper` / `industrial_metals` / `nickel` / `rubber` | Materials GICS | PALLFNFINDEXM / PCOPPUSDM / PZINCUSDM / PNICKUSDM / PRUBBUSDM |
| `energy_equities` / `thermal_coal` / `uranium` | Energy GICS | PNGASUSUSDM / PCOALAUUSDM / PURANUSDM |
| `softs_sugar` / `softs_cotton` / `softs_cocoa` / `softs_coffee` | Consumer Staples GICS | PSUGAISAUSDM / PCOTTINDUSDM / PCOCOUSDM / PCOFFOTMUSDM |

## Outputs

`macro_sector_shock.csv` — monthly (60y window):
`sector, date, basket_mom_12m, commodity_mom_12m, shock_score, shock_zone`

Reads: daily_prices.parquet / sector_prices.parquet,
sp500_constituents.parquet (GICS sector + sub-industry membership),
FRED CSV (cached under macro_data/, shared with the other macro scripts).

## Amplifier history (fetch_amplifier_history.py, 2026-08)

The focused subsector baskets name non-S&P amplifiers (TSM, ASML, SCCO,
AEM, BTI, ...). Their full OHLCV history is fetched from yfinance by
`fetch_amplifier_history.py` and appended to daily_prices.parquet
(+192,990 rows, 32 amplifiers, 1996-2026). Delisted names verified
unavailable: MRO/HES/CMA/SUM/CHX (HES acquired by CVX 2024, CHX delisted
2023) — the SECTORS table keeps them declared but they contribute nothing
until relisted; X (US Steel) delisted. The fetch script is the
single source of the amplifier set — re-runnable after a price refetch.

## Usage

```bash
python fetch_amplifier_history.py --max-years 30   # one-time amplifier add
python macro_sector_shock.py --save
```

Wired into `run_daily_automation.py` as the `taleb_sector_shock` job
(depends on `hmm`; feeds `export`).

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs
