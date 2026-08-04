# vol_target.py

**Volatility targeting** for SMCI (and the growth_ai sleeve).

## Rule


$$
w^* = \mathrm{clip}\!\left(\frac{\sigma_{\text{target}}}{\sigma_{\text{asset}}},\; w_{\min},\; w_{\max}\right)
$$


- $\sigma_{\text{asset}}$: annualized realized vol from log returns (default 21-day window)
- $\sigma_{\text{target}}$: desired *standalone* position vol (default 25%)
- **SMCI hard cap**: $w_{\max} = 5\%$ of portfolio
- Other growth_ai names: $w_{\max} = 8\%$

Optional portfolio-budget mode:


$$
w^* = \mathrm{clip}\!\left(\frac{\sigma_{\text{port}}\cdot \text{risk\_budget}}{\sigma_{\text{asset}}},\; w_{\min},\; w_{\max}\right)
$$


## Usage

```bash
# SMCI (default caps)
python vol_target.py --ticker SMCI --save

# Tighter target vol
python vol_target.py --ticker SMCI --target-vol 0.20 --w-max 0.05

# Spend 15% of a 12% portfolio-vol budget on SMCI
python vol_target.py --ticker SMCI --portfolio-vol 0.12 --risk-budget 0.15

# All growth_ai names (SMCI, NVDA, AMD, PLTR, CRWD)
python vol_target.py --growth-sleeve --save
```

## Outputs

- `vol_targets.csv` / `vol_targets.parquet` — weight/shares targets vs current holdings
- Prints TRIM/ADD share deltas and position-vol before → after

## Why SMCI is special

SMCI sits in the higher-risk growth/tech index and already appears in the personal portfolio. Vol targeting enforces a **small risk budget** regardless of inverse-vol math when $\sigma$ is moderate in-sample — the **5% weight cap** is the binding control.

Combine with fractional Kelly (`kelly.py`) and growth-index EW weights; vol targeting is the position-level risk governor.
