# cross_section.py

Multi-factor cross-section: rank the universe on value + quality + momentum,
long top quintile / short bottom quintile, monthly rebalance, sector-neutral —
point-in-time and OOS.

## Why it exists (rationale)

Single-name screens (preferred_metrics, peer_analytics) rank *against peers*;
this ranks the *whole universe* on a blended factor and trades the extremes.
Two properties make it honest:

1. **Point-in-time factors** — at each rebalance date, value/quality come from
   the most recent `fundamentals.parquet` row with `as_of_date <= rebalance`
   (no future fundamentals), and momentum is computed from price history up to
   that date (`mom_12_1` + `ret_21d`). No lookahead by construction.
2. **Sector-neutral** — each factor is percentile-ranked *within sector* first,
   the three ranks averaged, then quintiles of the average. Sector exposure is
   reported (`sector_exposure_abs_dev_avg`) as a check.
3. **OOS stats** — L/S portfolio vs equal-weight-long baseline via
   `cv_utils.oos_stats_vs_baseline`, only on days where factors exist.

## Usage

```bash
python cross_section.py --save
```

## Outputs

- `cross_section_rankings.csv` — family `cross_section`: `rebalance_date`
  (DATE), `ticker`, `bucket` (1=short, 5=long)
- `cross_section_returns.csv` — family `cross_section`: daily `long`, `short`,
  `long_short`, `equal_weight_long` returns
- `cross_section_stats.csv` — family `cross_section`: OOS stats vs baseline +
  `sector_exposure_abs_dev_avg`

## Related programs

- `cv_utils.py` — OOS stats vs baseline
- `momentum_analytics.py` / `factor_panel.py` — the factor ingredients this
  recomputes point-in-time
- `rebalance_calendar.py` — the monthly cadence this follows
- `preferred_metrics.py` — single-name screen; cross-section is the
  cross-sectional complement
