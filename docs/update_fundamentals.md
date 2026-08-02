# update_fundamentals.py

Refresh `fundamentals.parquet`: P/B, EV/EBITDA, market cap, total assets, mktcap_to_assets, etc.

## Purpose
Power value screens used for **portfolio inclusion** decisions (MOS vs CF, SHEL, FMC, trifecta).

## Decision thresholds (as used in analysis)
- Value trifecta: EV/EBITDA ≤ 9, P/B ≤ 1.5, MktCap/Assets ≤ 0.5
- Low EV/EBITDA and low P/B ranking tables in the dashboard

Keep this table current before re-running inclusion screens or the dashboard data export.
