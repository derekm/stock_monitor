# backtest_quick_ignition.py — 5-day-fractal quick ignition rules vs lagging momentum gate

## Why it exists (rationale)

The ride gate is a **LAGGING** momentum-level detector: `mom12>0.40` opens after a
surge, and the price-vs-momentum backtest showed all momentum/price-position
signals have **negative forward spread** (they capture the run too late → mean
reversion). This tests the opposite: the **5-DAY fractal (base-3 granularity,
(3,5) = 15-day full window)** can fire a **QUICK IGNITION** signal — price
re-accelerating on a volume surge — that catches a breakout **EARLY**, before
the longer momentum windows confirm.

## Signals (all DAILY OHLCV, no lookahead, position next day)

Quick ignition (finest fractal):
- `ign_vol_price` — 5d momentum turning up AND volume_z > 0.5
- `ign_breakout_vol` — close > trailing 5d high (short Donchian) AND volume_z > 0.5
- `ign_pctile_vol` — 5-day close_pctile > 0.8 AND volume_z > 0.5
- `ign_gap_vol` — open gap up AND volume_z > 0.5

Baseline:
- `mom_gate` — classic daily momentum gate (mom12>0.40 & mom3>0), exit mom3<=0

Risk managed by **ATR-chandelier stop (2×ATR)** and vol-scaled sizing from
ride-longevity work. Two size modes: `full` and `volscale`.

## Results (150 tickers, daily, no lookahead)

| rule | total_ride | mean_excess | hit_rate | mean_maxDD | in_market |
|------|-----------|-------------|----------|------------|-----------|
| buy_hold | +31,613 | 0.000 | — | — | 100% |
| **ign_gap_vol:full** | +522 | **−2.32** | 5.3% | −71% | 64% |
| ign_gap_vol:volscale | +499 | −2.47 | 7.3% | −62% | 68% |
| ign_pctile_vol:full | +383 | −3.24 | 3.3% | −67% | 51% |
| ign_breakout_vol:full | +347 | −3.48 | 3.3% | −66% | 47% |
| mom_gate:full | +225 | −4.29 | 2.0% | −54% | 26% |

**Verdict:** 5-day-fractal ignition signals **do not beat buy-hold**. The best quick rule
(`ign_gap_vol` = gap-up on volume surge) still trails BH by −2.3%/yr with 64%
in-market and 71% max drawdown. Mom gate is worst (−4.3% excess, 26% in-market).
All ignition rules fire too frequently on noise; the faster granularity doesn't
translate to better timing — it just increases trade frequency and drag.

The conclusion aligns with `backtest_price_vs_momentum.py`: **faster ≠ better for
entry timing** on this universe. The statistical edge comes from holding quality
rides, not from finer fractal granularity.