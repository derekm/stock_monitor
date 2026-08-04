# risk_parity_analytics.py

Analytics program that regenerates **vol targeting vs risk parity** comparison CSVs.

## Outputs

| File | Content |
|------|---------|
| `vol_target_vs_risk_parity.csv` | Personal portfolio: σ, current w, VT capped/renorm, RP inv-vol, RP ERC |
| `growth_ai_vol_vs_risk_parity.csv` | growth_ai sleeve comparison |

## Usage

```bash
python risk_parity_analytics.py
python risk_parity_analytics.py --window-vol 21 --window-cov 63
python risk_parity_analytics.py --smci-cap 0.05 --target-vol 0.25
python risk_parity_analytics.py --portfolio-only

# via analytics hub
python maintain_analytics.py vol-rp
python maintain_analytics.py all   # includes vol-rp
```

## Columns (portfolio)

- `sigma` — 21d annualized realized vol  
- `w_current` — holdings weight (normalized)  
- `w_VT_capped` — vol-target with per-name 5% / other caps  
- `w_VT_renorm` — capped VT renormalized to 100%  
- `w_RP_inv_vol` — diagonal risk parity $w \propto 1/\sigma$  
- `w_RP_ERC` — equal risk contribution using covariance  

See also [vol_target.md](vol_target.md).

## Related programs

- [docs/vol_target.md](vol_target.md)
- [docs/portfolio_optimization.md](portfolio_optimization.md)
- [docs/robust_covariance.md](robust_covariance.md)
- [docs/maintain_analytics.md](maintain_analytics.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
