# analytics_common.py

Shared Polars/pandas loaders, return helpers, and the **canonical constants**
used across the analytics programs. This is a **library**, not a runnable
script (no `argparse`, no `main()`).

## Why it exists (rationale)

Avoids every analytics script re-implementing the same parquet reads and return
math. It also centralizes the "prefer the clean prices copy" decision
(`daily_prices_clean.parquet` when present, else `daily_prices/`).

## Canonical constants (single source of truth — import, don't re-define)

- `BASE_THRESHOLDS` — the dual-pass / INCLUDE_CORE quality+value thresholds
  (`roe_min` 0.15, `roic_min` 0.15, `de_max` 1.0, `ev_max` 9.0, `pb_max` 1.5,
  `mca_max` 0.5). Consumers: `preferred_metrics.py`, `fundamentals_history.py`,
  `threshold_logic.py`, `inclusion_criteria.py`.
  Named aliases `ROE_MIN/ROIC_MIN/DE_MAX/EV_MAX/PB_MAX/MCA_MAX` for the older
  scripts.
- `quality_value_composite(...)` / `quality_value_parts(...)` — the weighted
  quality+value composite formula (q weights 0.35/0.35/0.15/0.15 on
  roe/roic/de/stability; v weights 0.4/0.3/0.3 on ev/pb/mca; composite
  0.55q + 0.45v). Replaces the copy-pasted formula that previously lived in
  `fundamentals_history.py` and `inclusion_criteria.py` — weights cannot drift.
- `HAS_POLARS` / `HAS_SCIPY` — capability flags (single import check).

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
