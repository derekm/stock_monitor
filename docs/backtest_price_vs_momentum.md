# backtest_price_vs_momentum.py — do PRICE-based fractal signals predict forward returns better than MOMENTUM-based ones?

## Why it exists (rationale)

The original fractal backtest (`fractal_windows_backtest_gpu.py`) tested only
momentum-consensus signals. Once the statistical profiler persists a wide set of
per-window statistics, the natural question is: **which family of statistics is
more predictive of future returns** — the price-position/price-shape family or
the momentum family? This script answers that by reading the SAVED profiles
(no recompute) and testing every feature against forward monthly returns.

## What it tests

Each feature is a 0/1 signal on the profile row at a window-end date, tested
against forward DAILY log returns over 21/63/126 trading days (1/3/6 months):

- **momentum** (existing): `log_ret>0`, `momentum>0`, `momentum>median`
- **price position**: `close_z>0`, `close>median`, `close>mean`, `close_z>0.5`,
  `close_pctile>0.5/0.8`, `runup>0.5/0.8`
- **price structure**: `price_skew>0`, `price_curvature>0`, `window_drawdown>-0.1`
- **volume**: `volume_z>0`, `close>vwap`, `vwap>median`
- **hybrid** (price+momentum): `pctile>0.5&mom>0`, `pctile>0.8&mom>0`,
  `runup>0.5&mom>0`

Per feature/horizon: `hit_rate_on`, `mean_on/off`, `spread`, `annual_spread`,
`base_mean`. The headline is the best signal per family on annualized spread —
answering the user's question "do price-based fractals work better than
momentum-based?"

## Result (583 tickers, daily forward 21/63/126 trading days, no lookahead)

| signal | horizon | hit_rate_on | annual_spread |
|--------|---------|-------------|---------------|
| vol:volume_z>0 | 21d | 0.559 | **+0.025** |
| vol:volume_z>0 | 63d | 0.584 | +0.002 |
| mom:momentum>0 | 126d | 0.599 | −0.020 |
| price:close>median | 126d | 0.598 | −0.023 |
| price:close_z>0 | 126d | 0.598 | −0.022 |
| struct:curv>0 | 63d | 0.575 | **−0.460** |

**Honest read:** price-based fractal signals do **NOT** beat momentum-based ones.
All signal families have negative annualized forward spread at 1-month horizon
(i.e. "on" predicts LOWER forward return than "off" — momentum/price-position
have already captured the run, so the forward leg mean-reverts). The ONLY
positive-spread signal is **volume** (`volume_z>0`, +0.025 at 21d) — a
volume-surge short-term edge. The worst are price-structure signals
(`curvature>0`, `drawdown>-0.1`), which strongly predict mean reversion
(−0.38 to −0.46 annualized). Price position is marginally better than momentum
at the 126d horizon but still negative.

## Usage

```bash
python backtest_price_vs_momentum.py                # horizons 21,63,126 trading days
python backtest_price_vs_momentum.py --horizon 21   # just 1-month forward
python backtest_price_vs_momentum.py --span 90      # only 90-day windows
```

Requires `fractal_profiles.parquet` (run `statistical_profiler.py --save` first).

## Outputs

- `backtest_price_vs_momentum.parquet` — per feature/horizon metrics, sorted by
  annualized spread, plus a printed best-per-family table.

## Related

- `statistical_profiler.py` — produces the profiles this reads
- `fractal_windows_backtest_gpu.py` — the momentum-only GPU universe backtest
