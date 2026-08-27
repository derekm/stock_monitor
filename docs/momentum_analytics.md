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

(Outputs are parquet; older CSV mirrors removed.)

(Schema family: screen_decision — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [factor_panel.md](factor_panel.md) — combines this
- [preferred_metrics.md](preferred_metrics.md)
- [buy_candidates.md](buy_candidates.md)
- [momentum_research.md](momentum_research.md) — the research measures
- [momentum_research_backtest.md](momentum_research_backtest.md) — confidence/backtest
