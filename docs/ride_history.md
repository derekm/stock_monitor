# ride_history.py

Point-in-time **recommended ride trade history** per ticker — reconstructs
what the ride rule would have said each month, with no lookahead, exactly as
`shock_ride.py` computes it.

## Why it exists (rationale)

`shock_ride_tickers.parquet` stores only the **latest** snapshot per ticker.
To audit a name's ride signals *over time* — when it entered, when it exited,
how many times the gate opened — you need the rule replayed month by month.
`ride_history.py` does that, faithful to the production decision logic, so
you can see a name's full recommended trade history instead of just today's
signal.

## Method (faithful to shock_ride.py)

For each monthly step (from the 3rd month of history forward), with **no
lookahead**:

- **mom3 / mom12** — trailing 3/12-month sum of monthly *log* returns
  (`resample('ME').sum()`, same as `_monthly_returns`).
- **fractal stack + posture** — 4-view ladder (15d/30d/45d/90d) on the daily
  close up to that month's end; `stack_depth` = consecutive confirmed spans.
- **long_ride_score** — durability (smoothness / pullback / overshoot /
  accumulation) on the daily series up to that month's end.
- **ride_gate** — quality entry, no 12-month history requirement.
- **ride_exit** — dual-condition exit (trailing_stop = −0.25).
- **recommendation** — via the SAME young/established branch as shock_ride:
  a ticker with < `MIN_TICKER_HISTORY` (36) months is **young** → `BUY` iff
  gate open (exit ignored); otherwise full dual-exit logic → `BUY` / `AVOID`
  / `WATCH` / `FLAT`.

## Usage

```bash
python ride_history.py --ticker RAL              # one ticker
python ride_history.py --ticker RAL,NVDA,XBI     # several
python ride_history.py --ticker RAL --save       # also write csv/parquet
```

## Outputs

- `ride_history.csv` / `ride_history.parquet` — `as_of, ticker, n_months,
  established, mom3, mom12, posture, stack_depth, long_ride_score,
  ride_gate_open, gate_horizon, gate_mom, ride_exit_flag, exit_kind,
  recommendation`

## Related programs

- `shock_ride.py` — the production daily ride screen (latest snapshot only)
- `ride_longevity.py` — `ride_gate` / `ride_exit` / `long_ride_score`
- `fractal_windows.py` — `momentum_stack` / `fractal_posture`
- `backtest_rides.py` — historical A/B of ride rule variants
