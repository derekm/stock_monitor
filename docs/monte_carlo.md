# monte_carlo.py

Regime-switching Monte Carlo with variance-reduction options.

## Why it exists (rationale)

Point forecasts understate tail risk. This simulates terminal wealth (and path
stats) by driving returns with the HMM transition matrix + regime-conditional
moments, and offers variance reduction (antithetic, control variate, stratified
initial regime, Sobol quasi-MC) so the estimates converge fast. It is the
quantitative backing for the tail-risk / drawdown views.

## Usage

```bash
python monte_carlo.py --index portfolio --n 5000 --horizon 252 --save
python monte_carlo.py --ticker AEP --antithetic --control --stratified
```

Flags: `--index`, `--ticker`, `--n` (paths), `--horizon`, `--antithetic`,
`--control`, `--stratified`, `--quasi`, `--save`. Reads `daily_prices.parquet`,
`hmm_regime_states.csv`, `hmm_transition_matrix.csv`.

## Outputs

- `monte_carlo_summary.csv` — percentile terminal wealth / path stats
- `monte_carlo_path_stats.csv` — per-path statistics
- `monte_carlo_terminal_wealth.csv` — terminal wealth distribution

(Schema family: forecast_anomaly — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [mcmc_regimes.md](mcmc_regimes.md) — regime-mean uncertainty input
- [hmm_regime_detection.md](hmm_regime_detection.md)
- [tail_risk_hedge.md](tail_risk_hedge.md) (if present)
