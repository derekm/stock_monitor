# vix_term_structure.py

VIX / volatility term-structure exploration using an offline realized-vol proxy.

## Why it exists (rationale)

The VIX term structure (contango vs backwardation) is a sentiment/risk signal,
but a live VIX feed isn't always available. This approximates the term structure
from realized vol at different horizons (from `daily_prices`) so the dashboard's
vol tab and the regime logic still get a slope signal offline.

## Usage

```bash
python vix_term_structure.py --save
```

Flags: `--save`. Reads `daily_prices.parquet`.

## Outputs

- `vix_term_structure.csv` — vol curve by horizon
- `vix_term_structure_summary.csv` — slope / contango summary

(Schema family: regime_state — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [kalman_state_estimates.md](kalman_state_estimates.md)
- [hmm_regime_detection.md](hmm_regime_detection.md)
- [regime_aware_constraints.md](regime_aware_constraints.md)
- [crisis_correlation.md](crisis_correlation.md)
