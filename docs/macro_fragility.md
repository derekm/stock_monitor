# macro_fragility.py

Macro debt-fragility layer (the Keen/Minsky findings) — the macro complement
to the micro `fragility_screen.py`.

## Why it exists (rationale)

The Taleb layer is micro: per-name fragility (leverage, tail alpha, gap
share). Steve Keen's work adds the MACRO half of the Financial Instability
Hypothesis: **aggregate private debt is the master fragility variable**.
The debt-to-GDP ratio grows cyclically then exponentially before crises
(Keen 1995/2013); the change in debt (ΔD) is the term in Keen's
aggregate-demand equation `E_t = Y_{t-1} + v·ΔD_t`. Palley (2014) critiques
the velocity form, but the debt-impulse-as-fragility insight survives —
Palley's own reduced form has bank credit ΔD₁ with unit impact on AD.

## Formulas

**All-sectors Debt / GDP ratio (quarterly):**

$$
\text{debt\_gdp\_ratio}(t) = \frac{\text{TCMDO}(t)}{\text{GDP}(t)}
$$

where TCMDO = FRED total credit market debt (all sectors, millions USD, 
divided by 1000 to billions); GDP = FRED nominal GDP (already annualized,
do NOT re-annualize).

**Debt impulse (Keen's ΔD/GDP, annualized):**

$$
\text{debt\_impulse}(t) = \frac{D(t) - D(t-4)}{\text{GDP}(t)}
$$

where $D(t)$ = TCMDO at quarter $t$; the 4-quarter change annualizes the
quarterly change to approximate "flow per year".

**Velocity-scaled impulse (Keen 2014 §9 — effective demand):**

$$
\text{debt\_impulse\_v}(t) = \text{debt\_impulse}(t) \times \text{M2V}(t)
$$

Keen 2014: $E = Y + v \cdot \Delta D$ — the true demand impact of a debt
change is LARGER than the bare change when M2 velocity $v > 1$.
Measured: 2007 peak bare 0.369 → v-scaled 0.729 (~2×).

**Credit Accelerator (Biggs-Mayer-Pick / Keen §13):**

$$
\text{debt\_acceleration}(t) = \frac{D(t) - 2D(t-1) + D(t-2)}{\text{GDP}(t)}
$$

Second difference of debt scaled by GDP — the Biggs-Mayer-Pick channel:
it's the CHANGE in debt growth that predicts unemployment, not the level.
Historical $r = +0.79$ vs house-price changes. Currently re-accelerating
(0.05-0.07 after 2023-24 near zero).

**Minsky Signal (stability breeds instability):**

$$
\text{minsky\_signal}(t) = \text{debt\_impulse}(t) \times (1 - p\_stress(t))
$$

Fragility accumulates DURING calm: when HMM stress posterior $p\_stress$
is low (tranquil regime), a high debt impulse means the system is quietly
levering. The signal is highest exactly when markets feel safest — validated:
its top quarters are 2007Q2-Q4 (debt building at ~37% of GDP/yr while
p(stress)=0, right before the GFC). A high impulse WITH high p_stress is
the crisis phase (deleveraging pressure), and the signal correctly
collapses.

**Minsky percentile:** rank of minsky_signal over full history (0-1).

**Danger zone (Keen 2009 AER thresholds):**

| Zone | Debt impulse | Interpretation |
|---|---|---|
| `benign` | < 5% | pre-1970 normal |
| `elevated` | 5–13% | 1987 counterfactual zone |
| `danger` | 13–20% | approaching crisis zone |
| `crisis_band` | ≥ 20% | 2008 trigger zone (deleveraging reduces demand, forces unemployment) |

Current reading: **crisis_band** (2025Q4 crossed from danger; 2007 peaks ~0.37,
well above 0.20 threshold). Since 1980: 77 crisis_band, 69 danger, 32
elevated, 1 benign quarters.

**Minsky signal percentile:** rank of minsky_signal over full history (0-1).

## Data

FRED public CSV endpoints (`fredgraph.csv?id=...`, no API key):

- `TCMDO` — total credit market debt, all sectors (quarterly, 1945-)
  UNITS: millions of dollars → divided by 1000 to billions.
- `GDP` — nominal GDP (quarterly, 1947-). UNITS: billions AND already at
  annual rate — do NOT re-annualize (rolling-sum inflates the ratio ~4×;
  caught against the known all-sectors credit-debt/GDP ≈ 3.6×).
- `M2V` — velocity of M2 money stock (quarterly, 1959-) for the
  velocity-scaled impulse.

Cached under `macro_data/`; refetched only when the last cached quarter is
stale (TTL 35d; FRED publishes with ~1 quarter lag).

## Outputs

`macro_fragility.csv` — quarterly (60y window):

```
date, debt_gdp_ratio, debt_impulse, debt_impulse_v, debt_acceleration,
velocity, p_stress, minsky_signal, minsky_pctile, danger_zone, regime_ctx
```

- `debt_gdp_ratio` — TCMDO / GDP (≈3.6 now, all-sectors)
- `debt_impulse` — YoY Δdebt / GDP (Keen's ΔD)
- `debt_impulse_v` — impulse × M2 velocity (Keen 2014 §9: E = Y + v·ΔD)
- `debt_acceleration` — Δ²(debt)/GDP (Keen §13 / Biggs-Mayer-Pick; r=+0.79
  vs house-price changes historically)
- `velocity` — M2 velocity (FRED M2V)
- `p_stress` — HMM stress posterior (same soft-stress belief as
  `buy_candidates.regime_stress_prob`), forward-filled quarterly
- `minsky_signal` — impulse × (1 − p_stress)
- `minsky_pctile` — rank of the signal over the full history
- `danger_zone` — Keen 2009 band label: <5% `benign` (pre-1970 normal),
  5-13% `elevated` (1987 counterfactual zone), 13-20% `danger`,
  ≥20% `crisis_band` (the 2008 trigger zone). Current: crisis_band.
- `regime_ctx` — HMM regime label at the quarter end

## Usage

```bash
python macro_fragility.py --save
```

Reads: FRED (network, cached), `hmm_regime_states.csv` (via buy_candidates).
Wired into `run_daily_automation.py` as the `taleb_minsky` job (depends on
`hmm` + `taleb_fragility`; feeds `export`).

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [macro_shock.md](macro_shock.md) — supply-shock twin (oil-only)
- [macro_sector_shock.md](macro_sector_shock.md) — sector/subsector extensions
- [shock_ride.md](shock_ride.md) — rides the explosions this layer labels
- [fragility_screen.md](fragility_screen.md) — micro fragility twin
- [hmm_regime_detection.md](hmm_regime_detection.md) — stress posterior source
- [export_dashboard_data.md](export_dashboard_data.md) — catalog + dashboard wiring