# pair_engine.py

Pair / relative-value engine: cointegration + residual mean-reversion inside
industry groups, with stops and time exits — walk-forward OOS.

## Why it exists (rationale)

The RF-style analysis exposed that sector peers co-move; the tradeable question
is whether a *spread* between two names in the same industry mean-reverts.
This engine answers it honestly:

1. **Universe** — pairs within `industry` (fallback `sector`), cap `--max-per-group` (default 40; SPX members first). Daily DAG uses `--n-folds 1`.
2. **Selection** — return-corr screen `|ρ| ∈ [corr_min, corr_max]`, then Engle-Granger + OU half-life, then BH-FDR. Usable = FDR-survive AND half-life ∈ [2, 250] days.
3. **Trading (next window)** — z-score entry ±2 (skip |z| > 6: relationship
   broke, not reversion), exit at z=0, stop at ±4, time exit after 60 bars.
   Position is beta-hedged: `pos × (ret_b − beta·ret_a)`.
4. **Walk-forward** — `--n-folds 3` non-overlapping OOS windows; selection ends
   the day before each test window starts. Stats aggregate OOS trades only.

## Usage

```bash
python pair_engine.py --save --max-per-group 40 --n-folds 1
python pair_engine.py --save --lookback 378 --test-days 252 --n-folds 3 \
    --entry-z 2.0 --exit-z 0.0 --stop-z 4.0 --max-hold 60 --alpha 0.20
```

## Outputs

- `pair_engine_pairs.csv` — family `pair_engine`: `pair_id`, `group`,
  `asset_a`, `asset_b`, `coint_t`, `p_value`, `beta`, `half_life`,
  `fdr_survive`, `usable`, `fold`, `z_now`
- `pair_engine_trades.csv` — family `pair_engine`: `pair_id`, `entry_date`,
  `exit_date` (DATE), `entry_z`, `exit_z`, `bars_held`, `exit_reason`
  (revert/stop/time), `hedged_pnl`, `z_pnl`, `fold`
- `pair_engine_stats.csv` — family `pair_engine`: `pair_id`, `n_trades`,
  `win_rate`, `avg_pnl`, `total_pnl`, `sharpe`, `folds`

## Related programs

- `cv_utils.py` — FDR + fold boundaries (the honest-OOS machinery)
- `peer_analytics.py` — the peer-group map this shares
- `allpairs_correlations.py` / `rolling_correlation_windows.py` — correlation
  context (regime-aware: pairs trade less in high_vol_stress)
