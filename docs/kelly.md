# kelly.py

Kelly criterion estimators for position sizing.

## Why it exists (rationale)

Sizing should be grounded in edge, not vibes. This computes full-Kelly (and
fractional) position sizes from either a continuous GBM form (stocks:
$f^* = (\mu - r)/\sigma^2$) or a binary edge form ($f^* = (bp - q)/b$), so the
sizing rules in `preferred_metrics` / `vol_target` can be cross-checked against a
Kelly-optimal stake.

## Usage

```bash
# Continuous (stocks)
python kelly.py --mu 0.13 --sigma 0.35 --r 0.04 --fraction 0.5
# Binary (edge)
python kelly.py --p 0.55 --b 1.0
# From stored params for a ticker
python kelly.py --ticker AEP --fraction 0.25
```

Flags: `--mu`, `--sigma`, `--r` (default 0.04), `--ticker`, `--fraction`,
`--p` (required for binary), `--b`, `--q`.

## Outputs

- `kelly_parameters.parquet` — stored per-ticker Kelly params (when `--ticker`
  with stored params)

(Schema family: summary_metrics — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [preferred_metrics.md](preferred_metrics.md) — sizing rules
- [vol_target.md](vol_target.md) — vol-target sizing
- [portfolio_optimization.md](portfolio_optimization.md)
