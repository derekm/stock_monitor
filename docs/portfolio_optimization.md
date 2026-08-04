# portfolio_optimization.py

**ERC risk parity** and **Global Minimum Variance (GMV)** portfolio construction.

## Equal Risk Contribution (ERC)

Each name contributes the same share of portfolio **variance**:


$$
RC_i = w_i\,(\Sigma w)_i = \frac{\sigma_p^2}{N}
$$


| Solver | Notes |
|--------|--------|
| **Multiplicative** | Fast; can drive some weights to 0 when correlations are awkward |
| **SLSQP + weight floor** | Preferred long-only ERC — equalizes RC with `w_i ≥ floor` (default 2%) |
| **Inverse-vol** | Diagonal approximation $w \propto 1/\sigma$; ignores correlations |

## Global Minimum Variance (GMV)


$$
\min_w \; w^\top \Sigma w \quad \text{s.t.} \quad \mathbf{1}^\top w = 1
$$


| Variant | Constraint |
|---------|------------|
| Unconstrained | Closed form $w \propto \Sigma^{-1}\mathbf{1}$ (shorts allowed) |
| Long-only | $w \ge 0$ via SLSQP |
| Long + per-name cap | $0 \le w_i \le c$, uniform cap $c$ (default 5%), all names |

GMV minimizes volatility; it does **not** equalize risk contributions (low-vol names get larger RC).

## Usage

```bash
python portfolio_optimization.py
python portfolio_optimization.py --universe portfolio --window 126 --name-cap 0.05
python portfolio_optimization.py --universe growth_ai --w-floor 0.05
python maintain_analytics.py optimize
```

## Outputs

- `erc_gmv_strategies.csv` — weight & RC by strategy × ticker  
- `erc_gmv_summary.csv` — portfolio vol, return, RC dispersion, max name weight  |

## How this differs from vol targeting

| Method | Objective |
|--------|-----------|
| **Vol targeting** | Cap / scale **one** name to a σ budget |
| **Inv-vol / ERC** | Balance risk **across** the book |
| **GMV** | Lowest achievable portfolio σ |

Practical stack: **GMV or ERC for core**, **vol-target cap on individual names** as a risk governor.

## Related programs

- [docs/vol_target.md](vol_target.md)
- [docs/risk_parity_analytics.md](risk_parity_analytics.md)
- [docs/robust_covariance.md](robust_covariance.md)
- [docs/kelly.md](kelly.md)
- [docs/black_litterman.md](black_litterman.md)
- [docs/regime_aware_constraints.md](regime_aware_constraints.md)
- [docs/maintain_analytics.md](maintain_analytics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
