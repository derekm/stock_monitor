# hmm_regime_detection.py

Gaussian HMM regime detection on market returns + realized vol + a correlation
proxy.

## Why it exists (rationale)

Regime is the master switch for risk posture (stress → tighter caps, fewer buys).
This fits a Gaussian HMM to daily market features and labels states post-hoc by
mean return / vol ordering (low_vol, normal, high_vol_stress) — the probabilistic
regime signal the rest of the stack consumes.

## Usage

```bash
python hmm_regime_detection.py --n-states 3 --save
```

Flags: `--n-states` (default 3), `--save`. Reads `daily_prices/`.

**Outputs:** `hmm_regime_states.parquet` (per-date regime label + state probabilities),
`hmm_regime_summary.csv`, `hmm_transition_matrix.csv`.

**Consumed by:** [regime_aware_constraints.md](regime_aware_constraints.md) (reads
`hmm_regime_states.parquet` directly; auto-runs this script if missing) and
[rebalance_calendar.md](rebalance_calendar.md) (reads `hmm_regime_states.parquet`).

## Outputs

- `hmm_regime_states.parquet` — per-date regime label + posterior
- `hmm_regime_summary.csv` — per-state mean return / vol
- `hmm_transition_matrix.csv` — Markov transition matrix

(Schema family: regime_state — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [hmm_posterior_analysis.md](hmm_posterior_analysis.md) — posterior exploration
- [regime_aware_constraints.md](regime_aware_constraints.md)
- [crisis_correlation.md](crisis_correlation.md) — alternative stress definition
- [kalman_regime.md](kalman_regime.md) (if present) / [vix_term_structure.md](vix_term_structure.md)
