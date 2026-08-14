# arista.py — ARISTA top-of-uptrend detector (DYNAMIC)

## What it does

For **every ticker** in the price universe, computes a daily point-in-time
**top-of-uptrend** detector — the companion to [shock_ride.md](shock_ride.md),
which gates *entry* on the slope of momentum and durability. ARISTA points the
same momentum lens at *exhaustion*: it flags when a strong uptrend's **slope is
rolling over at/near a high** — the exact setup that preceded FTNT's
Jul-9-2025 peak and its Aug-1-8 2025 breakdown.

It is a **top detector (de-risk / trim / stand down), not a timing tool**. On
FTNT the signal fired 7/7–7/9-2025, ~4 weeks before the break.

## Formulas

On split/dividend-adjusted closes (`adj_close`, via `load_adj_prices_pandas`),
all point-in-time (no lookahead):

$$
\text{mom}_3 = \frac{C_t}{C_{t-63}} - 1 \qquad
\text{mom}_6 = \frac{C_t}{C_{t-126}} - 1
$$

$$
\text{decel} = \text{mom}_6 - \text{mom}_3
$$

- $\text{decel} < 0$ → the 6-month trend slope is *less* than the 3-month
  slope = momentum decelerating / rolling over (the **leading** tell).

$$
\text{downshare} = \frac{\sum_{20d} \text{Vol}\cdot\max(0,-\Delta P)}
{\sum_{20d} \text{Vol}\cdot|\Delta P|}
\qquad
\text{from}_{20} = \frac{C_t}{\max_{20d} C} - 1
\qquad
\text{atYear} = \frac{C_t}{\max_{252d} C}
$$

- `downshare` rising toward/above 0.5 → distribution (sellers stepping in).
- `from20 < 0` → failing to make new highs (rollover).
- `atYear` near 1.0 → strong long-term trend context.

**ARISTA signal (the actionable top):**

$$
\text{signal} = \big(\text{atYear} > 0.92\big) \land \big(\text{decel} < -0.05\big)
$$

Momentum diverging while still within ~8% of the 1-yr high.

**ARISTA score (0–1 intensity):** three legs, each normalized to [0,1] with
fixed caps, combined, then gated by high-proximity:

$$
\text{div}_n = \min\left(\frac{-\text{decel}}{0.15},1\right)\qquad
\text{dist}_n = \min\left(\frac{\text{downshare}-0.5}{0.30},1\right)\qquad
\text{roll}_n = \min\left(\frac{-\text{from}_{20}}{0.12},1\right)
$$

$$
\text{score} = \text{atYear}\big|_{0.80}^{1.0} \cdot
\left(0.45\,\text{div}_n + 0.30\,\text{dist}_n + 0.25\,\text{roll}_n\right)
$$

So a healthy accelerating uptrend scores near 0; a genuine top (FTNT 7/2025:
decel −0.13, rollover starting, distribution rising) scores ~0.7–0.9.

## Honest measured results (full universe, signal → forward 120d)

| Metric | TOTAL |
|---|---|
| Signals | 274,189 across 592 tickers |
| Mean ARISTA score at signal | 0.35 |
| Avg forward max drawdown | **−24%** |
| Avg forward return | +7% |
| Caught ≥15% drawdown rate | **81%** |

IT/AI names (avg fwd maxDD, caught rate): AMAT −36% / 100%, CRWD −34% / 100%,
NET −42% / 100%, MU −40% / 100%, PANW −31% / 100%, NVDA −38% / 98%.

The signal is deliberately **broad** (it fires on any momentum divergence near
a high). Use the **score** to rank intensity and the **signal** as the de-risk
trigger; the backtest shows most signals are followed by a ≥15% drawdown within
120 sessions.

## Outputs

- `arista_metrics.parquet` — **full daily metrics time series per ticker**
  (the backtesting surface): `date, ticker, close, mom3, mom6, decel,
  downshare, from20, at_year_high, leg_divergence, leg_distribution,
  leg_rollover, arista_score, arista_signal`. Join on `(ticker, date)`.
- `arista_signals.parquet` — latest snapshot per ticker (date + last metrics +
  signal + interpretation).
- `arista_backtest.parquet` — per-ticker signal→forward-120d stats
  (`n_signals, mean_score, avg_fwd_dd, avg_fwd_ret, caught_rate`) + TOTAL.

## Usage

```bash
python arista.py --save                       # compute + write all outputs
python arista.py --tickers FTNT,PANW --save   # subset
python arista.py backtest --save              # aggregate backtest stats
```

Wired into `run_daily_automation.py` as `taleb_arista` (no deps — only reads
prices, runs at wave 0); feeds export. (Schema family: Taleb / fat tails — see
[SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [shock_ride.md](shock_ride.md) — the entry/ride gate this complements
- [ride_now.md](ride_now.md) — current-state recommendations
- [breakout_detector.md](breakout_detector.md) — FRESH_BREAKOUT/MATURING verdicts
- [fractal_windows.md](fractal_windows.md) — granularity-ladder momentum stack
- [subindustry_regime.md](subindustry_regime.md) — per-basket HMM stress
