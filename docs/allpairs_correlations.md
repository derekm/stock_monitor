# allpairs_correlations.py

Dense pairwise asset & sector correlations over rolling windows (long history + latest matrices).

```bash
python allpairs_correlations.py --window 63 --step 21 --max-assets 50
```

Outputs:
- `allpairs_asset_corr_history.csv` / `allpairs_sector_corr_history.csv` — rolling history
- `allpairs_asset_corr_latest.csv` / `allpairs_sector_corr_latest.csv` — latest matrices
- `allpairs_corr_summary.csv`
