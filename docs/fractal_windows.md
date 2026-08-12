# fractal_windows.py / fractal_windows_gpu.py — fractal sliding-window momentum

Implements the fractal-of-sliding-windows momentum scheme from patent
**US20120253946A1** (FIGS 26A/28/29) — the application I'm an inventor on.

## The scheme (from the patent)

A **fractal span** is a fixed window `(from, to)` within a total range `[0, a*b]`
where `a` = base span and `b` = repetitions. `spans_generator(a, b)` emits **every
aligned window** whose endpoints are multiples of `a`:

```
(30, 3) -> (0,30) (0,60) (0,90) (30,60) (30,90) (60,90)
```

That's every length that's a multiple of the base (30, 60, 90) at every aligned
start — the "fractal" of the range. FIG 29 slides each span forward over a
rolling offset (`past`), computing a statistical profile (here: momentum) on each
window `[past+from, past+to]`.

**Backward-looking framing (for momentum):** at rolling date `i`, each span `(x,y)`
covers the trailing window `[i-(y-x), i]`, so all fractal spans END at the current
date — the full multi-granularity decomposition of the trailing `a*b`-day range.

## Formulas

**Fractal span set** (FIG 28) — all aligned windows over $[0, a b]$:

$$
S(a,b) = (x, y) \quad x,y \in (0, a, 2a, ..., ab) \quad x < y
$$

**Trailing momentum return** over span `(x,y)` at date `i` (log-price $p$):

$$
r = p(i) - p(i - y + x)
$$

**Linear slope** of log-price over the trailing $L = y-x$ day window (closed form,
no polyfit), with $S_x$ and $S_{xx}$ the fixed sums of the day index and its square:

$$
slope = (L \cdot S_{xy} - S_x \cdot S_y) / (L \cdot S_{xx} - S_x^2)
$$

where $S_y$ and $S_{xy}$ are rolling sums of $p$ and $k \cdot p$ over the window
($k$ = global day index), computed via cumsum.

**Uptrend flag** — the span is trending up when both return and slope are positive:

$$
u = 1( r > 0 \land slope > 0 )
$$

**Risk-adjusted momentum** (return per unit volatility):

$$
M = r / \sigma
$$

**Fractal consensus** — fraction of spans in an uptrend and mean momentum at date
$i$ (over the $N = |S(a,b)|$ spans):

$$
\bar{u} = (1/N) \sum u
\qquad
\bar{M} = (1/N) \sum M
$$

A **breakout** fires when the majority of spans agree and momentum is strong:

$$
breakout = 1( \bar{u} \ge \theta_u \land \bar{M} \ge \theta_M )
$$

## Two engines, guaranteed identical

| | fractal_windows.py | fractal_windows_gpu.py |
|---|---|---|
| Within-ticker | vectorized pandas rolling (O(n), closed-form slope) | batched torch cumsum |
| Across-tickers | serial loop (parallelize with Pool) | one `[T x days]` tensor op |
| Consensus | polars group_by (fallback pandas) | on-device span-axis mean (no groupby) |

`test_fractal_cpu_gpu.py` proves they always concur (synthetic + real data, many
span configs). CPU is the safe fallback; GPU is the scatter-gather fast path.

## Backtest findings (full universe, GPU, 2026-08-11)

Fractal consensus (frac of spans in uptrend) is a **viable momentum signal** but
does NOT beat the best single window (30d) on annualized spread at any horizon:

| horizon | fractal frac>=0.6 | best single-window (mom_30d) |
|---------|-------------------|------------------------------|
| 3mo | +18.1% | **+21.3%** |
| 6mo | +9.9% | **+11.1%** |
| 12mo | +6.3% | **+6.8%** |

The combined `frac60_and_mom60` (fractal agreement ≥0.6 AND 60d momentum > 0) is
a close second at every horizon and more robust (higher hit rate). **Honest
conclusion:** fractal multi-granularity agreement is a real signal (hit rate
0.64-0.65) but not a free lunch over a single well-chosen window. Its strength is
**robustness and early detection for young tickers**, not raw predictive power.

## Files

- `fractal_windows.py` — spans_generator, fractal_signal_vec, fractal_consensus, breakout_score
- `fractal_windows_gpu.py` — fractal_batch (batched tensor), fractal_consensus_batch (on-device)
- `fractal_windows_backtest.py` — CPU parallel backtest (single-window vs fractal)
- `fractal_windows_backtest_gpu.py` — GPU scatter-gather backtest
- `test_fractal_cpu_gpu.py` — CPU/GPU concurrency test
- `run_tests.py` — executable test library (includes cpu_gpu, spans, fractal_vec)
