# analytics_common.py

Shared Polars/pandas loaders and return helpers used across the analytics
programs. This is a **library**, not a runnable script (no `argparse`, no
`main()`).

## Why it exists (rationale)

Avoids every analytics script re-implementing the same parquet reads and return
math. It also centralizes the "prefer the clean prices copy" decision
(`daily_prices_clean.parquet` when present, else `daily_prices.parquet`).

## Public functions

- `prices_path(prefer_clean=True)` → `Path` to the clean-or-raw prices parquet.
- `load_prices_pandas(prefer_clean=True, tickers=None)` → long `pd.DataFrame`
  with columns `date, ticker, close` (optionally filtered to `tickers`).
- `wide_closes(prices)` → date-indexed wide frame of closes (ticker columns).
- `simple_returns(wide)` → log/simple returns from a wide close frame.
- `clip_returns(rets, clip=0.35)` → returns clipped at ±`clip` (drops impossible moves).
- `load_membership()` → `monitored_stocks` frame.
- `load_preferred()` → `preferred_metrics` frame.
- `ann_stats(rets, rf=0.04)` → annualized stats dict (vol, ret, sharpe, …).

## Outputs

None (library). Reads `daily_prices*.parquet`, `monitored_stocks.parquet`,
`preferred_metrics.csv/.parquet`.

## Related programs

- Used by most analytics programs (e.g. [allpairs_correlations.md](allpairs_correlations.md),
  [crisis_correlation.md](crisis_correlation.md), [cross_asset_analysis.md](cross_asset_analysis.md)).
- Companion loader: [data_access.md](data_access.md).
