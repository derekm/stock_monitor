# posterior_entropy_dynamics.py

Dynamics of HMM posterior entropy — how regime uncertainty evolves and leads/lags
vol & correlation.

## Why it exists (rationale)

`hmm_posterior_analysis` measures uncertainty at each date; this tracks its
*dynamics*: normalized entropy over time, persistence of uncertain spells, and
lead/lag vs realized vol and correlation. Uncertainty that precedes vol spikes is
an early-warning signal; this quantifies that relationship.

## Usage

```bash
python posterior_entropy_dynamics.py --save
```

Flags: `--save`. Reads `hmm_posterior_analysis.csv` (falls back to
`hmm_regime_states.csv`).

## Outputs

- `posterior_entropy_dynamics.csv` — entropy series + lead/lag stats
- `posterior_entropy_summary.csv` — summary

(Schema family: regime_state — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [hmm_posterior_analysis.md](hmm_posterior_analysis.md) — entropy source
- [hmm_regime_detection.md](hmm_regime_detection.md)
- [kalman_state_estimates.md](kalman_state_estimates.md)
