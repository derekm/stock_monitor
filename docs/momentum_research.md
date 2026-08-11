# momentum_research.py

Research-grounded momentum measures for detecting (and detecting EARLY) momentum
based price explosions, before a full 12 months of history exists. Pure functions
(no data I/O) wired into `shock_ride.py` and `momentum_analytics.py`.

## Why it exists (rationale)

The classic 12-month ride gate (`12-1` momentum) is unavailable for newly listed /
recently onboarded tickers. The research audit identified measures that work on
shorter history, and a graduated young-ticker gate. This module encodes them.

## Measures implemented (with sources)

- **TSMOM (JFE 2012, Moskowitz-Ooi-Pedersen)** — `tsmom_signal()` / `tsmom_stats()`:
  sign of the past k-month return (k=3/6/12), optionally vol-scaled by 1/sigma.
  3-month lookback captures ~80% of the 12-month Sharpe.
- **JT 1993 (Jegadeesh-Titman)** — `jt_momentum()`: k-month formation / skip-month
  cross-sectional momentum ranking variable. 3/6/9/12-month formations all work.
- **STMOM (RFS 2022, Medhat-Schmeling)** — `stmom_1m()`: 1-month continuation among
  liquid high-turnover stocks. Works with a single month; reversal only dominates
  in illiquid microcaps (caller applies the liquidity filter).
- **52-week-high proximity (George-Hwang 2004)** — `gw52_high()`: nearness to the
  listing all-time-high predicts returns that DON'T reverse. Computable immediately
  for young names (the listing high IS the 52-week high).
- **First-month drop (Ritter 1991)** — `research_report()` drops the first ~1 month
  of history for young names, because the IPO pop/underpricing is a pricing
  phenomenon, not momentum.
- **Young-ticker gate** — `young_gate()`: graduated entry gate for <12-month names:
  >=6 mo clean history (strict min 3, first month dropped), annualized 3/6-mo
  momentum vs a maturity-scaled 40% gate, requires 6-mo>0 AND 1-mo>0 (RFS
  continuation) AND near-listing-high (GH anchor), with volatility + liquidity
  filters (short-term momentum only works in liquid high-turnover names).

## Usage

```python
from momentum_research import research_report, young_gate, tsmom_stats
m = monthly_log_returns(ticker)   # monthly log returns, DatetimeIndex
rep = research_report(m, annual_vol=vol, adv=adv, adv_series=adv_series)
yg = rep["young_gate"]            # gate_open, reliability, mom_3m_ann, ...
```

## Outputs

No files written directly. Produces the research-momentum columns consumed by
`shock_ride.py` (per-ticker ride) and `momentum_analytics.py` (momentum_metrics).

## Related

- `docs/momentum_analytics.md` — momentum_metrics columns
- `docs/shock_ride.md` — ride gate + young-ticker recommendation
- `docs/research_valuation_fragility_audit.md` — the source research
- `momentum_research_backtest.py` / `momentum_research_confidence.py` — backtest + confidence measures
