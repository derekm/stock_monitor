# mcmc_regimes.py

Lightweight MCMC for regime-conditional return means.

## Why it exists (rationale)

The stack already has point-estimate HMM labels + transitions. The first-order
driver of Monte-Carlo terminal-wealth dispersion (beyond path noise) is
parameter uncertainty in *within-regime means*. This runs a Gibbs/independent-MH
per-regime sampler to quantify that uncertainty, feeding the richer Monte-Carlo
in `monte_carlo.py`.

## Usage

```bash
python mcmc_regimes.py --index portfolio --save
python mcmc_regimes.py --ticker AEP,NVR --n-draw 2000 --burn 500 --seed 0
```

Flags: `--index` (repeatable), `--ticker`, `--n-draw` (default 2000),
`--burn` (default 500), `--seed` (default 0), `--save`. Reads
`hmm_regime_states.csv`.

## Outputs

- `mcmc_regime_means.csv` — posterior mean draws per regime/asset
- `mcmc_transition_draws.csv` — transition posterior draws
- `mcmc_regime_summary.csv` — posterior summaries

(Schema family: regime_state — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [hmm_regime_detection.md](hmm_regime_detection.md) — regime labels input
- [monte_carlo.md](monte_carlo.md) — consumes the uncertainty
- [kalman_state_estimates.md](kalman_state_estimates.md)
