# black_litterman.py

Black-Litterman expected returns & posterior weights.

## Why it exists (rationale)

Pure reverse-optimization + view-blending is a more disciplined way to set
expected returns than point estimates. It starts from equilibrium returns
(π = δ·Σ·w_mkt), blends investor views (P·μ = Q + ε), and produces posterior
returns μ_bl and mean-variance weights. It closes the loop from screens
(`preferred_metrics`) and regimes into a portfolio weight.

## Steps

1. Equilibrium returns π = δ Σ w_mkt (reverse optimization)
2. Investor views P μ = Q + ε, ε ~ N(0, Ω)
3. Posterior μ_bl = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹ π + P'Ω⁻¹ Q]
4. Mean-variance weights with μ_bl (long-only)

## Usage

```bash
python black_litterman.py --universe portfolio
python black_litterman.py --universe portfolio --view TICKER:0.05 --view OTHER:0.08
python black_litterman.py --universe growth --tau 0.05 --delta 2.5 --save
```

Flags (via `cli_common` + own): `--universe/--index`, `--ticker`, `--sector`,
`--view` (repeatable `TICKER:excess_return`), `--tau` (default 0.05),
`--delta` (risk aversion, default 2.5), `--window` (default 126), `--save`.

Views format: `TICKER:excess_return` (absolute view on a single asset). If no
`--view` is given, a few illustrative per-name views are used as examples.

## Outputs

- `black_litterman_weights.csv` — posterior weights per ticker

(Schema family: weights_performance — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [black_litterman_views.md](black_litterman_views.md) — builds views from screens/regimes
- [preferred_metrics.md](preferred_metrics.md) — source of conviction
- [portfolio_optimization.md](portfolio_optimization.md) — ERC/GMV alternative
- [robust_covariance.md](robust_covariance.md) — Σ input
