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

## Data

FRED public CSV endpoints (`fredgraph.csv?id=...`, no API key):

- `TCMDO` — total credit market debt, all sectors (quarterly, 1945-)
  UNITS: millions of dollars → divided by 1000 to billions.
- `GDP` — nominal GDP (quarterly, 1947-). UNITS: billions AND already at
  annual rate — do NOT re-annualize (rolling-sum inflates the ratio ~4×;
  caught against the known all-sectors credit-debt/GDP ≈ 3.6×).

Cached under `macro_data/`; refetched only when the last cached quarter is
stale (TTL 35d; FRED publishes with ~1 quarter lag).

## Outputs

`macro_fragility.csv` — quarterly (60y window):
`date, debt_gdp_ratio, debt_impulse, p_stress, minsky_signal,
 minsky_pctile, regime_ctx`

- `debt_gdp_ratio` — TCMDO / GDP (≈3.6 now, all-sectors)
- `debt_impulse` — YoY Δdebt / GDP
- `p_stress` — HMM stress posterior (same soft-stress belief as
  `buy_candidates.regime_stress_prob`), forward-filled quarterly
- `minsky_signal` — impulse × (1 − p_stress)
- `minsky_pctile` — rank of the signal over the full history

## Usage

```bash
python macro_fragility.py --save
```

Reads: FRED (network, cached), `hmm_regime_states.csv` (via buy_candidates).
Wired into `run_daily_automation.py` as the `taleb_minsky` job (depends on
`hmm` + `taleb_fragility`; feeds `export`).

(Schema family: Taleb / fat tails — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs
