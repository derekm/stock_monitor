# hmm_posterior_analysis.py

Explores the HMM hidden-state posterior probabilities produced by
`hmm_regime_detection.py`.

## Why it exists (rationale)

Hard regime labels hide uncertainty. This script reads `hmm_regime_states.csv`
and reports posterior mass over time, entropy (mixed-belief days), soft vs hard
labels, and transition risk when the max posterior is weak — so the dashboard
and risk logic can treat regime as a probability, not a boolean.

## Usage

```bash
python hmm_posterior_analysis.py --save
```

Flags: `--save`. If `hmm_regime_states.csv` is missing it runs
`hmm_regime_detection.py --save` first.

## Outputs

- `hmm_posterior_analysis.csv` — posterior mass by regime over time
- `hmm_uncertain_days.csv` — days with weak/mixed posterior
- `hmm_posterior_summary.csv` — per-regime summary

(Schema family: regime_state — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [hmm_regime_detection.md](hmm_regime_detection.md) — the states it reads
- [regime_aware_constraints.md](regime_aware_constraints.md)
- [buy_candidates.md](buy_candidates.md) / [black_litterman_views.md](black_litterman_views.md)
