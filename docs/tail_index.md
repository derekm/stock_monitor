# tail_index.py — Fat-tail diagnostics

## Description
Measures the ACTUAL tail behavior of the price history, where the Gaussian
assumption understates risk by orders of magnitude (the Taleb layer).

## Why it exists (rationale)
Sharpe ratios, variance, and Gaussian Monte Carlo all assume thin tails. A
tail index of α < 3 means variance is nearly meaningless as a risk measure —
the portfolio's fate is decided in the tails. This script quantifies how
wrong the Gaussian lens is, per name and for the equal-weight portfolio.

## Usage
```
python tail_index.py [--tickers A,B,C] [--top-pairs 30]
```

## Outputs (see SCHEMAS → `taleb` family)
- `tail_index.csv` — per-ticker Hill tail index α, empirical vs Gaussian
  P(|z|>3σ/5σ), tail ratio (how many × more 5σ events than Gaussian), kurtosis.
- `portfolio_tail.csv` — same metrics for the equal-weight portfolio.
- `tail_dependence.csv` — pairwise upper/lower tail dependence (P(both in the
  q-quantile tail)/q) for the 80 most-liquid names, top pairs by max dependence.

## Related
gap_risk.py (the other half of the tail lens), fragility_screen.py (consumes
tail_alpha_hill), barbell_check.py. Registered as the `taleb_tail` daily job.
