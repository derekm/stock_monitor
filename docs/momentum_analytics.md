# momentum_analytics.py

Cross-sectional and time-series momentum metrics per ticker, plus quintile
spreads and IC vs forward return.

## Why it exists (rationale)

Momentum is one of the factor inputs to `factor_panel` / `preferred_metrics` and
the buy decision. This computes trailing-horizon returns (21/63/126/252d), a
time-series momentum score (z-scored horizons averaged), the classic 12-1
skip-month momentum, and residual momentum vs the market — then ranks
cross-sectionally and tests IC.

Since the research audit it also emits the research-grounded momentum measures
from [`momentum_research.py`](momentum_research.md): TSMOM 3/6/12 Sharpe
(JFE 2012), JT-6 (JT 1993), STMOM-1 (RFS 2022), GW-52w-high proximity
(George-Hwang 2004), and the graduated **young-ticker gate** (Ritter first-month
drop, vol/liquidity filters) so newly listed tickers get an early-momentum read.

## Usage

```bash
python momentum_analytics.py --universe all --save
python momentum_analytics.py --universe portfolio
```

Flags: `--universe` (index list or `all`, default `all`), `--save`. Reads
`daily_prices/`.

## Outputs

- `momentum_metrics.parquet` — per-ticker momentum metrics (incl. research columns:
  `tsmom_3mo_sharpe`, `tsmom_6mo_sharpe`, `tsmom_12mo_sharpe`, `jt_6_1_ret`,
  `stmom_1m_ret`, `gw52_high_prox`, `mom_3m_ann`, `mom_6m_ann`, `young_gate_open`,
  `young_gate_reliability`, `signal_age_months`)
- `momentum_quintiles.parquet` — cross-sectional quintile spreads
- `momentum_ic.parquet` — IC vs forward 21d return
- `momentum_jt.parquet` — Phase 2 item 8 long-short comparison (annualized
  net/gross per signal, overlap tape) — see below

## Phase 2 item 8 — JT 12-2 vs 12-1 vs 12-month fractals (measured 2026-09-01)

Construction: monthly-rebalance EW top/bottom-quintile long-short, 10 bps/side,
448-month full tape (PIT signals at month-end, hold one month, vs TMI).

| signal | full-window geometry | skip | net ann. |
|--------|----------------------|------|----------|
| `mom_12_1` | 252d return, skip 21d (existing) | 21d | **+16.4%** |
| `mom_12_2` | 252d return, skip 42d (paper replica) | 42d | **+17.8%** |
| `mom_fractal_12_b3_21` | stack (21,3)(42,3)(63,3)(84,3) = 63/126/189/252d | 21d | +11.0% |
| `mom_fractal_12_b3_42` | same | 42d | +11.8% |
| `mom_fractal_12_b6_21` | single view (42,6) = 2-month bars × 6 | 21d | +10.9% |
| `mom_fractal_12_b6_42` | same | 42d | **+12.1%** |

Verdict: **12-2 beats 12-1 by +1.3 pp/yr — below the +2 pp bar (FAIL).** The
longer skip helps (12-2 > 12-1), and finer granularity helps a little (b6_42
edges b3_42 by +0.3 pp), but no 12-month fractal clears the JT pair (best
fractal − best JT = **−5.6 pp**). **The fractal stack is a ride tool, not a
Jegadeesh/Titman replica** — its 15–90d breadth semantics do not carry the
12-month premium. Skip semantics: windows end `skip` trading days before the
signal date (JT's drop-the-recent-month), matching 12-1/12-2.

(Outputs are parquet; older CSV mirrors removed.)

(Schema family: screen_decision — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [factor_panel.md](factor_panel.md) — combines this
- [preferred_metrics.md](preferred_metrics.md)
- [buy_candidates.md](buy_candidates.md)
- [momentum_research.md](momentum_research.md) — the research measures
- [momentum_research_backtest.md](momentum_research_backtest.md) — confidence/backtest
