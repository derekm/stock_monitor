# kalman_state_estimates.py

Kalman filter latent-state estimates for market risk: a smooth latent return
level + stochastic-vol proxy, plus a smoothed latent correlation factor.

## Why it exists (rationale)

Regime labels are noisy day-to-day; a Kalman filter gives a smooth,
probabilistically-weighted latent state for market return and vol (and average
pairwise correlation). That smoothed state is a cleaner risk signal than raw
daily prints and feeds the dashboard's regime tab.

## Usage

```bash
python kalman_state_estimates.py --save
```

Flags: `--save`. Reads `daily_prices/`, `hmm_regime_states.csv`.

## Outputs

- `kalman_state_estimates.csv` — latent return / vol / corr over time
- `kalman_state_summary.csv` — per-state summary

(Schema family: regime_state — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [kalman_gain_analysis.md](kalman_gain_analysis.md) — gain diagnostics
- [hmm_regime_detection.md](hmm_regime_detection.md)
- [vix_term_structure.md](vix_term_structure.md)
- [regime_aware_constraints.md](regime_aware_constraints.md)
