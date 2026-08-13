# rare_ignition_info.py — does a price run add information quality/value miss?

## Setup

Quality/value (`preferred_metrics`) pick the **business**. They update on
filings. In a live market the first bar you can actually buy is a **tape**
bar. Question: does a rare volume-backed run carry information besides
“everyone already piled in”?

T+1 forward log excess vs equal-weight universe. Full 583-name price tape.

## Buckets

- `gap_vol_raw` — every gap-up + 20d volume z > 0.5
- `fresh_only` — no prior gap_vol in 63 sessions
- `rare_quality` — fresh + close > SMA200 + **not** extended (<15% above SMA200)
- `exuberant_fresh` — fresh + close > SMA200 + **already** extended (≥15%)
- `rare_laggard` — rare_quality and 63d residual vs EW < 0

`rare_quality` was the anti-exuberance hypothesis.

## Measured

| bucket | fires | 21d ann xs | 63d ann xs | 252d ann xs | 252d hit |
|--------|------:|-----------:|-----------:|------------:|---------:|
| gap_vol_raw | 523,623 | +0.6% | +0.2% | +0.05% | 0.50 |
| fresh_only | 3,136 | +2.4% | +0.7% | +0.15% | 0.51 |
| **rare_quality** | 967 | **−3.6%** | **−3.2%** | **−0.8%** | 0.49 |
| **exuberant_fresh** | **345** | +2.3% | **+7.4%** | **+2.6%** | **0.56** |
| rare_laggard | 466 | −0.4% | −0.3% | 0.0% | 0.49 |

The anti-exuberance filter **loses**. The tape’s information is the opposite:
a **first-in-63-day gap+volume print on a name already ≥15% above its 200-day**.
That is continuation of a sponsored run (near-high + paid-up size), not a
quiet name waking up.

~345 such events in the whole history ≈ a handful per year across 583 names.
That is what “rare” actually means.

## Live (as-of 2026-08-12, last 63 sessions)

Only two `exuberant_fresh` prints: **SPCX** (Jun 12) and **SATS** (Jul 17).
Both preferred **AVOID**, ride **FLAT**. Neither is a quality/value include,
and neither is a ride BUY. The tape-only set right now is not a long-hold
add list.

## How to use it

Do **not** replace quality/value with ignition. Quality/value still pick
**who** can be owned for a decade.

Use `exuberant_fresh` as a **when to add capital** overlay on names you
already tolerate — ideally BROAD + stack 4/4 + BUY/WATCH. It is not a
first-bar screen for cheap businesses. The “first bar” of a long hold is
still the quality/value (or ride-quality) decision; the rare print is
permission to size in *after* the run is already on.

```bash
python rare_ignition_info.py
```

Outputs: `rare_ignition_info.parquet`, `rare_ignition_live.parquet`.
Research-only.
