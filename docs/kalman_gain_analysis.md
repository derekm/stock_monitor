# kalman_gain_analysis.py

Kalman gain path analysis for the (market return, log-vol) filter.

## Why it exists (rationale)

The Kalman filter's gain sequence shows *how much* new observations are trusted
vs the prior state over time — a diagnostic for the regime/state estimates. This
script runs the filter (local level + stochastic vol proxy) and traces the gain
path so you can see when the filter is anchored vs reactive.

## Usage

```bash
python kalman_gain_analysis.py --save
```

Flags: `--save`. Reads `daily_prices/`, `hmm_regime_states.csv`.

## Outputs

- `kalman_gain_path.csv` — gain path over time
- `kalman_gain_summary.csv` — summary stats

(Schema family: regime_state — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [kalman_state_estimates.md](kalman_state_estimates.md) — the filter it runs
- [hmm_regime_detection.md](hmm_regime_detection.md) — regime input
- [vix_term_structure.md](vix_term_structure.md)
