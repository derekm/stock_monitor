# sobol_qmc.py

Sobol' quasi-Monte Carlo sequences for Gaussian shocks.

## Why it exists (rationale)

Monte-Carlo convergence is faster with low-discrepancy sequences than i.i.d.
uniform draws. This maps Sobol' points (even fill of [0,1]^d) through Φ⁻¹ to
quasi-Gaussian samples for MC integration — the `quasi` variance-reduction path
used by `monte_carlo.py`. Not a pipeline script; a numerical utility.

## Usage

```bash
python sobol_qmc.py        # demo / self-test of the sequence + inverse-CDF mapping
```

Flags: minimal (see source). Primarily an importable helper.

## Outputs

None written to disk (demo prints to stdout). Used by `monte_carlo.py`.

## Related programs

- [monte_carlo.md](monte_carlo.md) — consumes the quasi-Gaussian draws
- [mcmc_regimes.md](mcmc_regimes.md)
