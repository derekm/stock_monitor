# update_fundamentals.py

Refresh `fundamentals.parquet`: P/B, EV/EBITDA, market cap, total assets, mktcap_to_assets, etc.

## Purpose
Power value screens used for **portfolio inclusion** decisions (MOS vs CF, SHEL, FMC, trifecta).

## Decision thresholds (as used in analysis)
- Value trifecta: EV/EBITDA ≤ 9, P/B ≤ 1.5, MktCap/Assets ≤ 0.5
- Low EV/EBITDA and low P/B ranking tables in the dashboard

Keep this table current before re-running inclusion screens or the dashboard data export.

## Related programs

- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/fundamentals_history.md](fundamentals_history.md)
- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/dupont_analysis.md](dupont_analysis.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
