# allpairs_correlations.py

Dense pairwise asset & sector correlations over rolling windows — long history
plus the latest matrices.

## Why it exists (rationale)

Broad pairwise correlation is the basic input to diversification and regime
analysis: it shows how correlated the book is, how that changes over time, and
which names drive the latest correlation cluster. It feeds `crisis_correlation`
(calm vs stress) and the dashboard's correlation views.

## Usage

```bash
python allpairs_correlations.py --window 63 --step 21 --max-assets 60
```

Flags (own argparse, not `cli_common`):

- `--window` — rolling window length in trading days (default 63)
- `--step` — stride between windows (default 21)
- `--max-assets` — cap on number of assets to include (default 60)

## Outputs

- `allpairs_asset_corr_history.csv` — rolling asset correlation over time
- `allpairs_sector_corr_history.csv` — rolling sector correlation over time
- `allpairs_asset_corr_latest.csv` — latest asset correlation matrix
- `allpairs_sector_corr_latest.csv` — latest sector correlation matrix
- `allpairs_corr_summary.csv` — summary of correlation levels

(Schema family: correlation matrix — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [crisis_correlation.md](crisis_correlation.md) — calm vs crisis breakdown
- [cross_asset_analysis.md](cross_asset_analysis.md) — cross-asset/sector
- [maintain_analytics.md](maintain_analytics.md) — `all` regenerates these
