# Known issues

Verified against source. Open items only — resolved items are not listed here.

## rebalance_calendar.csv is not consumed by any script

`rebalance_calendar.py` writes `rebalance_calendar.csv` (scheduled rebalance dates +
regime context), but no downstream script reads it.

The scripts that act on regime —
[regime_aware_constraints.py](docs/regime_aware_constraints.md),
[portfolio_optimization.py](docs/portfolio_optimization.md),
[vol_target.py](docs/vol_target.md) — read the regime label from
`hmm_regime_states.csv` (or prices/holdings) directly, not the calendar output.

Effect: the calendar's `action` (`full_rebalance` / `reduced_rebalance`) and
`turnover_band` are computed but do not change any rebalancing or sizing decision.
The calendar is currently a reporting artifact.

Fix options (not yet implemented): have `portfolio_optimization.py` /
`vol_target.py` read `rebalance_calendar.csv` and apply `turnover_band` as a cap on
weight drift during `high_vol_stress`, or fold the calendar's `action` into
`regime_aware_constraints.py`'s policy table.
