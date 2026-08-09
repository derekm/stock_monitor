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

Two findings implemented:

1. **Debt impulse** — Δ(private debt)/GDP, annualized. Debt growing faster
   than GDP = the economy levering up = fragility accumulating. Keen's
   DebtWatch variable: US private debt went ~150% → ~300% of GDP 1980-2008,
   the Great Moderation masking the build-up.

2. **Minsky signal ("stability breeds instability")** — debt impulse ×
   (1 − p(stress)). Fragility accumulates DURING calm: when the HMM stress
   posterior is low (tranquil regime), a high debt impulse means the system
   is quietly levering. The signal is highest exactly when markets feel
   safest — validated: its top quarters are 2007Q2-Q4 (debt building at
   ~37% of GDP/yr while p(stress)=0, right before the GFC). A high impulse
   WITH high p(stress) is the crisis phase (deleveraging pressure), and the
   signal correctly collapses.

3. **Velocity-scaled impulse** — debt_impulse_v = impulse × M2 velocity.
   Keen 2014 §9: effective demand = income + velocity × Δdebt; the demand
   impact of a debt change is LARGER than the bare change (velocity ~1.4-2).
   Measured: 2007 peak bare impulse 0.369 → v-scaled 0.729 (~2×).

4. **Credit Accelerator** — debt_acceleration = Δ²(debt)/GDP (Keen §13 /
   Biggs-Mayer-Pick). The acceleration channel is distinct from the level:
   historical r = +0.79 vs house-price changes. Currently re-accelerating
   (0.05-0.07 after 2023-24 near zero).

5. **Danger zone** — debt impulse labelled by Keen's 2009 thresholds:
   <5% benign (pre-1970), 5-13% elevated (1987 near-miss), 13-20% danger,
   ≥20% crisis_band (2008 trigger zone). Current reading: **crisis_band**
   (2025Q4 crossed from danger; the 2007 precursor peaks sat ~0.37, well
   above the 0.20 threshold). Since 1980 the US has spent 77 quarters in
   crisis_band, 69 in danger, 32 elevated, 1 benign — the structural rise
   of debt-financed demand is Keen's central empirical claim.

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
`date, debt_gdp_ratio, debt_impulse, debt_impulse_v, debt_acceleration,
 velocity, p_stress, minsky_signal, minsky_pctile, regime_ctx`

- `debt_gdp_ratio` — TCMDO / GDP (≈3.6 now, all-sectors)
- `debt_impulse` — YoY Δdebt / GDP (Keen's ΔD)
- `debt_impulse_v` — impulse × M2 velocity (Keen 2014 §9: E = Y + v·ΔD —
  the true demand impact of a debt change; velocity ~1.4-2.0 makes it
  larger than the bare change)
- `debt_acceleration` — Δ²(debt)/GDP (Keen §13 / Biggs-Mayer-Pick Credit
  Accelerator; r=+0.79 vs house-price changes historically)
- `velocity` — M2 velocity (FRED M2V)
- `p_stress` — HMM stress posterior (same soft-stress belief as
  `buy_candidates.regime_stress_prob`), forward-filled quarterly
- `minsky_signal` — impulse × (1 − p_stress)
- `minsky_pctile` — rank of the signal over the full history
- `danger_zone` — Keen 2009 band label: <5% `benign` (pre-1970 normal),
  5-13% `elevated` (1987 counterfactual zone), 13-20% `danger`,
  ≥20% `crisis_band` (the 2008 trigger zone; deleveraging reduces demand
  and forces unemployment). Current reading: crisis_band.

## Usage

```bash
python macro_fragility.py --save
```

Reads: FRED (network, cached), `hmm_regime_states.csv` (via buy_candidates).
Wired into `run_daily_automation.py` as the `taleb_minsky` job (depends on
`hmm` + `taleb_fragility`; feeds `export`).

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs
