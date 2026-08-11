# momentum_research_backtest.py + momentum_research_confidence.py

Backtest harnesses for the research-grounded momentum measures in
[`momentum_research.py`](momentum_research.md).

## Why they exist (rationale)

The measures are only useful if they actually separate winners from losers. These
two scripts test that honestly across the full price universe and historical depth,
and find usable confidence measures.

## momentum_research_backtest.py — single-measure predictive power

For every ticker and month, records whether each signal is ON and the forward
3/6/12-month log return. Aggregates to hit-rate and mean forward return for
signal-on vs signal-off, plus an annualized spread.

**Measures tested:** TSMOM 3/6/12 (JFE 2012), JT-6 (JT 1993), STMOM-1 (RFS 2022),
GW-52w high (George-Hwang 2004), and a volatility-cap filter.

```bash
python momentum_research_backtest.py [--tickers N]
```

## momentum_research_confidence.py — signal-agreement as confidence

Rebuilds the same matrix and buckets by the NUMBER of momentum signals on
(1..6), testing the hypothesis that *agreement* raises reliability. This is the
confidence measure for entries.

```bash
python momentum_research_confidence.py [--tickers N]
```

## Honest findings (full universe + history, 2026-08-11)

### Single measures — all work, modestly
Every momentum measure has positive predictive power vs signal-off:

| Measure | hit_rate_on | annualized spread |
|---------|------------|-------------------|
| TSMOM-3 | ~0.60 | ~+7% |
| TSMOM-6 | ~0.63 | ~+7.5% |
| TSMOM-12 | ~0.60-0.67 | ~+7% |
| JT-6 | ~0.63-0.67 | ~+7.5% |
| GW-52w-high | ~0.60-0.67 | ~+7% |
| STMOM-1 | ~0.59-0.67 | ~+6% |

The measures are individually weak-to-moderate (hit rates 0.60-0.67) but all
positive — consistent with the academic literature.

### Confidence: signal-agreement does NOT monotonically raise reliability
The key honest result — hit-rate by number of agreeing signals is roughly FLAT
(not monotone):

| horizon | hit rate (1..6 signals) |
|---------|-------------------------|
| 3mo | 0.597, 0.601, 0.593, 0.603, 0.602, 0.599 |
| 6mo | 0.615, 0.626, 0.619, 0.634, 0.636, 0.634 |
| 12mo | 0.645, 0.656, 0.656, 0.667, 0.673, 0.671 |

**Implications:**
- **The measures are correlated** (they all proxy the same momentum factor), so
  stacking them does NOT add confidence. Requiring 6 signals ≈ requiring 1.
- **The confidence lever is HORIZON, not agreement**: hit-rate rises from ~0.60
  (3mo) to ~0.67 (12mo). Longer holds are more reliable — matches the momentum-
  persistence research (peak ~6-12mo, partial reversal beyond).
- **A 0.60-0.67 hit rate is a modest edge, not a knife-edge.** Size accordingly;
  do not treat any single momentum signal as high-confidence.

## Outputs

- `momentum_research_backtest.parquet` — per-feature predictive-power table
- `momentum_research_confidence.parquet` — hit-rate by signal-count per horizon

## Related

- [`momentum_research.md`](momentum_research.md) — the measures
- `docs/research_valuation_fragility_audit.md` — the source research
