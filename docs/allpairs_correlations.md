# allpairs_correlations.py

Dense pairwise asset & sector correlations over rolling windows (long history + latest matrices).

```bash
python allpairs_correlations.py --window 63 --step 21 --max-assets 50
```

Outputs:
- `allpairs_asset_corr_history.csv` / `allpairs_sector_corr_history.csv` — rolling history
- `allpairs_asset_corr_latest.csv` / `allpairs_sector_corr_latest.csv` — latest matrices
- `allpairs_corr_summary.csv`

## Related programs

- [docs/cross_asset_analysis.md](cross_asset_analysis.md)
- [docs/crisis_correlation.md](crisis_correlation.md)
- [docs/maintain_analytics.md](maintain_analytics.md)
- correlation_stability_metrics(in maintain_analytics)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
