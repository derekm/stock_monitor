# build_index.py

Build the **fertilizer / agrochemical equal-weight index** levels and membership artifacts.

## Purpose
Track the fertilizer sleeve (MOS, CF, NTR, …) used in backtests and sector-rotation analysis.

## Outputs
- Index levels (often into `index_levels_1y.parquet` / related tables)
- Membership consistent with `monitored_stocks.index_member`

## Related
- `build_defensive_index.py` for the defensive value sleeve
- `maintain_analytics.py backtest` for performance vs personal portfolio
