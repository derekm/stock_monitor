# signal_aggregator.py

Combine the five signal families into one per-ticker composite with
OOS-derived weights.

## Why it exists (rationale)

Five independent engines — `preferred_metrics` (single-name screen),
`peer_analytics` (relative vs group), `cross_section` (cross-sectional rank),
`pair_engine` (relative-value z), `earnings_catalyst` (event timing) — all
score the same universe but were never combined. This is the aggregation layer:
it normalizes each family to [0,1], estimates each family's information
coefficient (rank corr of score vs forward 21d return) on a **trailing window
ending before the live point**, and composites with IC-derived weights
(negative-IC families contribute zero).

## Usage

```bash
python signal_aggregator.py --save
python signal_aggregator.py --save --cutoff 2026-07-31
```

## Outputs

- `signal_aggregator_scores.csv` — family `aggregate`: `ticker`, per-family
  normalized scores (`preferred`,`peer`,`cross`,`pair`,`earnings`),
  `composite`, `rank`
- `signal_aggregator_ic.csv` — family `aggregate`: `family`, `ic`, `n`,
  `weight`, `weight_norm` (the only numbers worth quoting)

## Related programs

- The five family producers listed above (its inputs)
- `cv_utils.py` — trailing-window discipline (IC estimated at cutoff − 21d)
- `cost_model.py` — composites feed cost-aware backtests, not the reverse
