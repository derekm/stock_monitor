# earnings_catalyst.py

Earnings catalyst filter: pre-earnings momentum + post-earnings drift stats +
IV-vs-realized flag (realized-vol proxy).

## Why it exists (rationale)

Earnings are the highest-conviction catalyst for single-name moves. This
program turns the raw `earnings_calendar.parquet` into a decision input:

1. **Pre-earnings momentum** — 21d return as a percentile of the ticker's own
   252d rolling return distribution. Top tercile = `hot`, bottom = `cold`.
2. **Post-earnings drift (PEAD)** — historical surprise buckets (big_beat ≥+5%,
   beat 0–5%, miss <0%) → forward 5/20/63d returns. Drift is computed from a
   **trailing window only** (`--drift-window`, default 750 days) ending at
   `--cutoff`, so expected-drift for live signals is never fitted on the rows
   it scores (same discipline as pass-5).
3. **IV vs realized** — no options feed in the repo yet, so the proxy is the
   ratio of 21d realized vol to 63d realized vol. Ratio > 1.2 → `iv_rich`
   (options priced rich vs the stock's own vol base). A real IV feed can slot
   in behind the same column.
4. **catalyst_score** — hot pre-mom +1, cold −0.5, iv-rich −0.5, positive
   expected 20d drift +0.5. Sorted descending = most attractive earnings setup.

## Usage

```bash
python earnings_catalyst.py --save
python earnings_catalyst.py --save --cutoff 2026-07-31 --drift-window 500 --lookback 21
```

## Outputs

- `earnings_catalyst_signals.csv` — SCHEMAS family `earnings`:
  `ticker`, `next_earnings_date` (DATE), `surprise_pct`, `pre_mom_pctile`,
  `pre_mom_flag`, `iv_vs_realized`, `iv_rich`, `expected_drift_20d`,
  `catalyst_score`
- `earnings_drift_stats.csv` — family `earnings`:
  `bucket`, `n_events`, `drift_5d`, `drift_20d`, `drift_63d`

## Related programs

- `update_earnings.py` — populates `earnings_calendar.parquet` (its input)
- `buy_candidates.py` / `preferred_metrics.py` — natural consumers of the flag
- `cv_utils.py` — trailing-window discipline enforced here
