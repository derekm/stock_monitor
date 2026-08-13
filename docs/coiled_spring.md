# coiled_spring.py — BB/KC squeeze + shakeout → expansion detector

## What it detects

A "coiled spring" setup: prolonged BB/KC squeeze, low BB width percentile, a high-volume shakeout below the lower band, reclaim, and (optionally) early expansion confirmation.

## Signals

1. `squeeze_active` — BB inside KC for ≥10 of last 20 sessions
2. `width_compressed` — BB width at ≤25th percentile vs 252-day lookback
3. `shakeout_day` — close below BB lower band AND volume_z ≥ 1.5 in last 20 days
4. `reclaimed` — close back inside BB within 5 days of shakeout
5. `expand_confirmed` — BB width expanded ≥20% from shakeout low
6. `fund_*` — quarterly fundamentals (EV/EBITDA, ROIC, D/E, interest coverage, earnings stability) forward-filled to daily

## FTNT case study (Aug 15, 2025 – Apr 15, 2026)

| period | BB width | state |
|--------|---------:|-------|
| Dec 2024 – Aug 2025 | 0.12 → **0.55** | **Unwind/stretch** — coil released downward (Aug 7 earnings: −22% on 7.6× vol) |
| Aug 2025 – Apr 2026 | **0.55 → 0.10** | **Re-compression** — BB/KC squeeze 15/20 days |
| Apr 10, 2026 | vol_z **1.98**, BB pos **−0.07** | Shakeout / test of lows |
| May 7, 2026 | Earnings → **+35% in 3 weeks** | Violent expansion (+100% in 10 weeks) |

The 8 months *before* Aug 15, 2025 was the **unwind** (spring stretched), not compression. The compression happened *after* the earnings gap-down.

## Universe scan (asof 2026-04-15)

- 579 names scanned
- 82 squeeze active, 80 width-compressed, 118 shakeout, 115 reclaimed, 39 expanded
- **FTNT is the only INCLUDE_QUALITY name with shakeout + reclaim + not yet expanded**

## Usage

```bash
python coiled_spring.py --ticker FTNT --asof 2026-04-15
python coiled_spring.py --universe --asof 2026-04-15
```

Outputs: `coiled_spring_screen.parquet`
Research-only.
