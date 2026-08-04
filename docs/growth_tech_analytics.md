# growth_tech_analytics.py

Full analysis suite for the **growth / tech index** (`growth_tech_index` flag).

## What it runs

| Analysis | Output |
|----------|--------|
| Membership / sleeves | `growth_tech_membership.csv` |
| Vol & returns | `growth_tech_vol_returns.csv` |
| Correlation matrix | `growth_tech_correlation_matrix.csv` |
| Sleeve-average corrs | `growth_tech_sleeve_correlations.csv` |
| Rolling pairwise corr | `growth_tech_rolling_corr.csv` |
| Corr stability | `growth_tech_corr_stability.csv` |
| Sleeve performance | `growth_tech_sleeve_performance.csv` |
| Index levels vs peers | `growth_tech_index_levels_compare.csv` |
| Backtest stats | `growth_tech_backtest_stats.csv` |
| ERC / GMV / VT weights | `growth_tech_risk_models.csv` |

Related:
```bash
python portfolio_optimization.py --universe growth
python fisher_index.py --universe growth_tech --save
python forecast_granite.py forecast --index growth --horizon 10
python vol_target.py --growth-sleeve --save
```

## Usage

```bash
python growth_tech_analytics.py
python growth_tech_analytics.py --window 126
python maintain_analytics.py growth-tech
```

## Related programs

- [docs/build_growth_tech_index.md](build_growth_tech_index.md)
- [docs/index_registry.md](index_registry.md)
- [docs/allpairs_correlations.md](allpairs_correlations.md)
- [docs/cross_asset_analysis.md](cross_asset_analysis.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
