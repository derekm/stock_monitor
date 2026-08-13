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

## Second-generation entry gate: `structural_gate`

`ride_gate` is a **lagging momentum-level detector** — it opens after a surge
(mom12>thresh), buys the top, then holds through the pullback (the young path
suppresses the exit). On volatile / young / whipsaw names (e.g. RAL) it loses to
buy-and-hold. `structural_gate` responds to the **price/risk structure** instead
of a lagging momentum level. `structural_positions(close, mode=...)` returns a
daily position series (0/partial/full); `structural_gate(close, mode=...)` gives
the current signal.

### Modes (all daily, no lookahead)
- **`turtle`** — Donchian 55-day breakout entry + 2×ATR chandelier trailing stop
- **`volscale`** — exposure sized to target annualized vol (default 0.30), gated
  by SMA200 trend
- **`regime`** — EMA50/EMA200 markup/distribution state machine
- **`recouple`** — enter when close re-couples above EMA21 AND EMA50, size by 1/vol
- **`momentum`** — the classic daily momentum gate (mom12>0.40 & mom3>0, exit
  mom3≤0), included for comparison
- **`hybrid`** — momentum entry + vol-scaled size + 2×ATR chandelier stop
  (best drawdown control across the universe backtest)
- **`consensus`** — majority of the four structural signals, vol-scaled size

## Backtest evidence (`backtest_structural.py`, 250 tickers, daily, no lookahead)

| paradigm | total ride | mean excess | hit rate | mean maxDD |
|----------|-----------|-------------|----------|-----------|
| buy_hold (reference) | 1403.8 | 0.00 | 1.00 | −0.748 |
| volscale | 727.6 | −2.70 | 0.028 | −0.621 |
| consensus | 543.6 | −3.44 | 0.012 | −0.608 |
| recouple | 508.9 | −3.58 | 0.008 | −0.627 |
| regime | 507.8 | −3.58 | 0.004 | −0.644 |
| momentum | 344.5 | −4.24 | 0.008 | −0.524 |
| hybrid | 221.6 | −4.73 | 0.004 | **−0.433** |
| turtle | 82.1 | −5.29 | 0.008 | −0.564 |

**Honest conclusions:**
- **No timing approach beats buy-and-hold on raw return** across the broad
  universe — long-horizon buy-and-hold captures the big secular winners.
- **The gate's real value is drawdown control:** hybrid cuts max drawdown to
  **−0.433 vs −0.748 buy-hold** (42% reduction); momentum −0.524; turtle −0.564.
  Buy-hold's ~75% average max drawdown is the risk the gate exists to avoid.
- **Best risk-adjusted (Calmar):** volscale 1172 > consensus 894 > recouple 812.
- On **young / high-beta / whipsaw names (e.g. RAL)**, the structural modes beat
  the momentum gate on RAL: volscale +32.8%, regime +17.7%, momentum +16.5% vs
  the momentum-threshold gate's −19% — but still under buy-hold (+50.5%).

## Related

- `fractal_windows.py` — `momentum_stack` / `momentum_stack_series` (the stack
  the gate and exit consume)
- `shock_ride.py` — the per-ticker ride that uses the gate + dual exit
- `backtest_rides.py` — the historical A/B test of ride approaches
- `momentum_research.py` — the young-gate this quality gate generalizes
