# ride_longevity.py — early detection of breakouts that become LONG rides

## Why it exists (rationale)

The classic ride rule (enter on 12m momentum, exit on 3m rollover) is blunt: it
enters every breakout and exits on the first short-term dip. Two failure modes
dominate — **premature exit** (a single weak month inside a durable trend kicks
you out and you miss the rest of the ride) and **lost gains** (a real breakdown
is ridden too long because 3m stays positive). This module adds a durability
layer that separates breakouts that *run* from breakouts that *fake out*, a
quality gate that opens rides without requiring 12 months of history, and a
dual-condition exit.

## Why it exists (research anchors)

- **Vol-scaled momentum beats raw momentum** (Griffin, Ji & Martin) — smooth,
  low-vol advances persist; choppy spikes mean-revert.
- **Daniel & Moskowitz (2016)** — crash risk concentrates in high-vol, deep-
  drawdown momentum states; a volatility/smoothness filter trims that tail.
- **George & Hwang (2004) near-high** — names that hold near their 52-wk high
  have persistent continuation; pullback resilience extends rides.
- **Trend durability** — institutional accumulation (OBV) and shallow pullbacks
  distinguish sponsored trends from multiple-expansion pops.

## Functions

### `long_ride_score(close, volume=None, *, window=60, fundamentals=None)`
Composite 0-1 durability score (per date) for a long, durable ride:
- **smoothness** (0.30) — low-vol advance persistence (`|mean ret| / std`)
- **pullback_depth** (0.20) — shallow pullbacks inside the advance
- **not over-extended** (0.20) — price not stretched far above its trend
- **volume accumulation** (0.15) — sustained OBV up-slope (institutional)
- **fundamental support** (0.15) — durable earnings if provided, else neutral

A high score near a breakout = smooth, holds pullbacks, room to run, backed by
volume → the ride will run. Used to gate **ride extension** (hold past a dip).

### `ride_gate(m, *, entry_thresh=0.40, stack_depth=0, long_ride=0.0, reliability)`
Quality-based ride ENTRY gate that does **not** require 12 months of history.
Uses the LONGEST momentum horizon actually available (12mo→6mo→3mo ladder) and
uses **signal quality** (fractal stack depth + durability) as the confidence
substitute history length used to provide:
- ≥12mo: light bar (`stack_depth≥1` OR `long_ride≥0.35`)
- ≥6mo: medium bar (`stack_depth≥2` OR `long_ride≥0.40`)
- ≥3mo: strict bar (`stack_depth≥3` AND `long_ride≥0.45`)
All horizons require 1m continuation. A 4-month clean durable multi-granular
breakout opens where it previously could not.

### `ride_exit(m, *, exit_thresh=0, stack_depth=0, long_ride=0.0, trailing_stop=None, persist=1)`
Dual-condition ride-OVER test — exits only on a **confirmed** breakdown:
- `exit_soft` = 3m mom ≤ 0 (short-term rollover)
- `confirm` = stack collapsed (≤1 views) OR durability low (<0.35)
- exit only when rollover AND confirmation both hold (or a hard trailing stop)
A strong ride can dip a month without its stack collapsing → a negative 3m month
with an intact stack is a **pullback (hold)**, not an exit.

## Backtest evidence (`backtest_rides.py`, 250 tickers, 1-mo signal→trade lag)

| strategy | total ride | mean excess | hit rate | mean maxDD | trades |
|----------|-----------|-------------|----------|-----------|--------|
| classic (baseline) | 819.8 | −0.17 | 0.42 | −0.208 | 6199 |
| **quality_gate** | **837.2** | −0.10 | 0.39 | −0.203 | 6172 |
| dual_exit (alone) | 752.9 | −0.44 | 0.33 | −0.130 | 10052 |
| **quality_dual** | 819.2 | −0.18 | 0.41 | **−0.104** | 9322 |
| quality_dual_persist | 749.2 | −0.46 | 0.34 | −0.187 | 8280 |

**Honest conclusions:**
- **quality_gate** raises return (+17 over baseline) — removing the 12mo
  requirement lets durable young breakouts in early without lowering quality.
- **quality_dual** (quality entry + dual exit) matches baseline return while
  **halving max drawdown** (−0.104 vs −0.208) — the best risk-adjusted approach.
- **dual_exit alone** churns (10k trades) and gives back return — the exit needs
  the quality gate's better entries to pay off.
- **persistence (2-mo confirm)** *hurts* (−0.46 excess) — delaying exits on real
  breakdowns costs more than the whipsaw it avoids. **Do not use persist.**

## Outputs

No standalone output file; consumed by `shock_ride.py` which emits:
`long_ride_score`, `ride_gate_open`, `ride_gate_horizon`, `ride_gate_mom`,
`ride_exit_flag`, `ride_exit_kind` per ticker.

## Related

- `fractal_windows.py` — `momentum_stack` / `momentum_stack_series` (the stack
  the gate and exit consume)
- `shock_ride.py` — the per-ticker ride that uses the gate + dual exit
- `backtest_rides.py` — the historical A/B test of ride approaches
- `momentum_research.py` — the young-gate this quality gate generalizes
