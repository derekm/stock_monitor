# vol_target.py

**Volatility targeting** for any single name (a representative example is the growth_ai sleeve).

## Rule

$$
w^* = \mathrm{clip}\!\left(\frac{\sigma_{target}}{\sigma_{asset}},\; w_{\min},\; w_{\max}\right)
$$

- $\sigma_{asset}$: annualized realized vol from log returns (default 21-day window)
- $\sigma_{target}$: desired *standalone* position vol (default 25%)
- $w_{\max}$: per-name weight cap (default 5%; higher-risk names may use 8%)

Optional portfolio-budget mode:

$$
w^* = \mathrm{clip}\!\left(\frac{\sigma_{port}\cdot risk budget}{\sigma_{asset}},\; w_{\min},\; w_{\max}\right)
$$

## Usage

```bash
# Example name with default caps
python vol_target.py --ticker NVDA --save

# Tighter target vol
python vol_target.py --ticker NVDA --target-vol 0.20 --w-max 0.05

# Spend 15% of a 12% portfolio-vol budget on one name
python vol_target.py --ticker NVDA --portfolio-vol 0.12 --risk-budget 0.15

# All growth_ai names (NVDA, AMD, PLTR, CRWD)
python vol_target.py --growth-sleeve --save
```

## Outputs

- `vol_targets.csv` / `vol_targets.parquet` — weight/shares targets vs current holdings
- Prints TRIM/ADD share deltas and position-vol before → after

## Why vol targeting

Vol targeting enforces a **small risk budget** per name regardless of what inverse-vol sizing alone would imply when $\sigma$ is moderate in-sample — the **weight cap** is the binding control.

Combine with fractional Kelly (`kelly.py`) and growth-index EW weights; vol targeting is the position-level risk governor.

## Related programs

- [docs/portfolio_optimization.md](portfolio_optimization.md)
- [docs/risk_parity_analytics.md](risk_parity_analytics.md)
- [docs/kelly.md](kelly.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
