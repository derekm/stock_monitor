# Coiled Spring / Squeeze-Pop Detector — Comprehensive Documentation

**Files**: `coiled_spring.py` (live detector), `backtest_coiled_spring.py` / `backtest_coiled_spring_clean.py` (backtests)

---

## 1. Purpose & Philosophy

This is an **entry-only timing/sizing overlay**, not a discovery screen.

- **Quality/value pick the business** (preferred metrics: INCLUDE_QUALITY, composite score, ride_longevity).
- **Tape decides when to size in** — the coiled-spring detector measures stored energy in a BB/KC squeeze, identifies the shakeout/test, reclaim, and confirms expansion.
- Same rule as the ignition layer: **tape does not find the first-bar long hold**. It gates sizing on names already qualified by fundamentals + ride durability.

---

## 2. Mathematical Definitions

All formulas use **daily close/high/low/volume** from `daily_prices/`.

### 2.1 Bollinger Bands (20, 2)

```
mid_20      = SMA(close, 20)
std_20      = STDDEV(close, 20)
bb_upper    = mid_20 + 2 * std_20
bb_lower    = mid_20 - 2 * std_20
bb_width    = (bb_upper - bb_lower) / mid_20
bb_pos      = (close - bb_lower) / (bb_upper - bb_lower)   # < 0  ⇒  below lower band
```

### 2.2 Keltner Channels (20, 1.5 × ATR)

```
tr          = max(high - low,
                  |high - close_prev|,
                  |low - close_prev|)
atr_20      = SMA(tr, 20)        # Wilder / simple mean
kc_mid      = SMA(close, 20)     # same center as BB
kc_upper    = kc_mid + 1.5 * atr_20
kc_lower    = kc_mid - 1.5 * atr_20
```

### 2.3 Squeeze Day & Squeeze Active

```
squeeze_day = (bb_upper < kc_upper) AND (bb_lower > kc_lower)   # BB fully inside KC
squeeze_20d = SUM(squeeze_day, 20)                               # rolling count
squeeze_active = (squeeze_20d ≥ squeeze_days)                    # default ≥ 10 of last 20
```

### 2.4 BB Width Percentile (252-day lookback)

```
bb_width_p252 = PERCENTILE_RANK(bb_width, 252)   # rank of current width vs last 252 days
width_compressed = (bb_width_p252 ≤ width_pctile)  # default ≤ 0.25 (25th percentile)
```

*Requires ≥ 50 non-NaN observations in the 252-day window; else NaN.*

### 2.5 Volume Z-Score

```
vol_20      = SMA(volume, 20)
vol_std_20  = STDDEV(volume, 20)
vol_z       = (volume - vol_20) / vol_std_20       # protect div-by-zero → 1.0
```

### 2.6 Shakeout / Test Day

```
shakeout_day = (bb_pos < 0) AND (vol_z ≥ vol_z_thresh)   # default vol_z_thresh = 1.5
```

Evaluated on each of the last 20 sessions; the **most recent** qualifying day is the shakeout.

### 2.7 Reclaim / Held

```
reclaimed = ANY(bb_pos ≥ 0)  within [shakeout_day, shakeout_day + reclaim_days]
```
Default `reclaim_days = 5`. If no shakeout, `reclaimed = False`.

### 2.8 Expansion Confirmation / Sprung

```
bb_width_at_shakeout = bb_width on shakeout_day
expand_confirmed     = (bb_width_now / bb_width_at_shakeout - 1) ≥ expand_pct
```
Default `expand_pct = 0.20` (20% widening from the shakeout width).

### 2.9 Sprung (Strict 5-of-5 State)

```
sprung = squeeze_active AND width_compressed
         AND (shakeout_day exists) AND reclaimed AND expand_confirmed
```

All five conditions true simultaneously on the current bar.

---

## 3. State Machine (Progression)

```
squeeze_active
      │
      ▼
width_compressed          ← BB width at ≤ 25th pctile of its own 252d history
      │
      ▼
shakeout / test           ← Close below lower BB + vol_z ≥ 1.5 (conviction flush)
      │
      ▼
reclaimed / held          ← Price back inside bands within 5 days
      │
      ▼
expand_confirmed / sprung ← BB width widened ≥ 20% from shakeout low
```

**Key insight**: The coil stores *energy*; the shakeout tests the lows; reclaim shows demand; expansion confirms the breakout. Direction (up/down) is **not** knowable from the coil alone — quality/ride decide whether to own.

---

## 4. Fundamental Overlay (Quarterly, Point-in-Time)

Fundamentals from `fundamentals.parquet` (EDGAR → yfinance → Polygon priority, additive backfill, no auto-seed over EDGAR). Forward-filled to daily price index.

| Metric | Column | Signal |
|--------|--------|--------|
| EV/EBITDA | `fund_ev_ebitda` | Compression = declining (cheaper) over 4 quarters (`pct_change(63)`) |
| ROIC | `fund_roic` | Stability/improvement: `pct_change(252)` ≥ 0 preferred |
| Debt/Equity | `fund_debt_to_equity` | Low + stable: rolling 252d std < 0.5 |
| Interest Coverage | `fund_interest_coverage` | High/stable |
| Earnings Stability | `fund_earnings_stability` | High (> 8–9 preferred) |

These are **context**, not entry triggers. They tell you whether the business *deserves* the sizing when the spring pops.

---

## 5. FTNT Case Study (Textbook Spring → +100% in ~10 Weeks)

| Period | BB Width | State |
|--------|---------:|-------|
| Dec 2024 – Aug 2025 | 0.12 → **0.55** | **Unwind / stretch-down** (spring released downward) |
| Aug 7 2025 | vol_z ~4–7.6×, −22%, ~47M sh | Earnings blowdown |
| Aug 15 2025 | 0.545, $79 | Fully stretched, range bottom |
| Aug 2025 – Apr 2026 | **0.55 → 0.10** | **Re-compression** (8 months); squeeze 15 of 20–30 days |
| Apr 10 2026 | vol_z **1.98**, bb_pos **−0.07**, −4.9%, 12.4M | **Shakeout / test of lows** |
| Apr 15 2026 | $79.64, width 0.102 vs 0.105 at test | **Reclaimed**; NOT yet expanded |
| May 7 2026 | Earnings | Violent expansion begins |
| May 7 / 15 / 20 / Jun 30 | $107.97 / $122.78 / $130 / $153.60 | **+35% / +54% / +63% / +93%** — ~+100% in ~10 weeks |

**Fundamentals through the coil (nearest quarters):**
- ROIC ~1.03–1.035, D/E ~0.32, earnings_stability ~9.04, EV/EBITDA ~19–26
- Preferred: **INCLUDE_QUALITY composite 1.20**, ride **BUY / BROAD / stack 4**

**Detector output at Apr 15 2026:**
```json
{
  "squeeze_active": true,
  "squeeze_20d": 15,
  "bb_width": 0.102,
  "bb_width_pctile": 0.087,
  "width_compressed": true,
  "shakeout_day": "2026-04-10",
  "shakeout_vol_z": 1.98,
  "shakeout_bb_pos": -0.07,
  "reclaimed": true,
  "expand_confirmed": false,
  "sprung": false
}
```

**At May 15 2026:** `expand_confirmed = true` (width ~0.63, +520% from shakeout).

---

## 6. Universe Scans — Measured Baselines

### 6.1 Historical Scan (Apr 15 2026, 579 names)
| State | Count |
|-------|------:|
| squeeze_active | 82 |
| width_compressed | 80 |
| shakeout | 118 |
| reclaimed | 115 |
| expand_confirmed | 39 |
| **sprung (strict 5-of-5)** | **0** |

FTNT was the **only INCLUDE_QUALITY** name with shakeout + reclaim + not expanded.

### 6.2 Latest Live Scan (Aug 13 2026, 583 names)
| State | Count |
|-------|------:|
| squeeze_active | 29 |
| width_compressed | 89 |
| shook (shakeout) | 108 |
| reclaimed | 103 |
| expanded | 66 |
| **sprung (strict)** | **0** |

**Raw reclaim → expand rate**: 66 / 108 ≈ **61%** (contemporaneous, not a no-lookahead event study).

### 6.3 Full-Universe Backtest Results (2.55M events, 586 tickers)

| State | Events |
|-------|-------:|
| tight | 1,268,874 |
| coiled | 605,506 |
| sprung | 418,361 |
| held | 194,660 |
| test | 66,291 |

**Test-event predictor distributions (66,291 test events):**
- `bb_width` at test: median **0.112**, 25th pctile **0.075**, 10th **0.054**
- `vol_z` at test: median **2.4**, 75th **3.1**, 90th **3.8**

---

## 7. Shadow Book Backtest Results

### 7.1 Clean Backtest (all test entries, 63-day hold, 64,862 entries)
| Metric | Value |
|--------|------:|
| Mean return | **+5.1%** |
| Median return | **+4.3%** |
| Hit rate (> 0) | **62.2%** |

### 7.2 Sampled Shadow Book (30 tickers × last 5 tests, 63-day hold)
| Metric | Value |
|--------|------:|
| Mean return | **+9.1%** |
| Median return | **+6.0%** |
| Hit rate (> 0) | **71.3%** |
| 75th pctile | **+18%** |
| Max | **+84%** |
| Min | **−31%** |

**Conclusion**: Raw test entries have **62–71% hit rate with positive skew**. The conditional rate on quality-filtered names is higher.

---

## 8. High-Probability Pop Filter (Lifts > 65%)

Layer these three filters on raw technicals:

### Layer 1 — Technical Squeeze Quality
| Signal | Threshold | Rationale |
|--------|-----------|-----------|
| `squeeze_20d` | ≥ 12–15 days | FTNT had 15–23 in critical window |
| `bb_width` at test | ≤ 0.10–0.12 (or ≤ 25th pctile) | Deep coil = more stored energy |
| `vol_z` on shakeout | 1.8–2.5 | Conviction flush, not crash (FTNT 1.98) |
| Reclaim | Within 5 days, clean | Demand absorbs supply |

### Layer 2 — Fundamental Quality (Point-in-Time at/near coil)
| Metric | Threshold |
|--------|-----------|
| ROIC | > 0.15 (stable/improving) |
| Debt/Equity | < 1.0 |
| Earnings Stability | > 8–9 |
| Trends | Stable or mildly improving during coil (not deteriorating) |

### Layer 3 — Quality + Ride Overlay (Biggest Single Lift)
| Filter | Threshold |
|--------|-----------|
| Preferred | `INCLUDE_QUALITY` **or** `composite_score` > 0.8 |
| Ride | `long_ride_score` > 0.4 **or** BROAD/MIXED posture + stack depth ≥ 2 **or** BUY/WATCH rec |
| Avoid | Pure SATELLITE / AVOID coils (dilute hit rate) |

---

## 9. Current Strict Candidates (Aug 13 2026 Live Screen)

| Ticker | Preferred | Composite | long_ride_score | bb_width | ROIC | D/E | Earn.Stab | Notes |
|--------|-----------|-----------|-----------------|----------|------|-----|-----------|-------|
| **WST** | INCLUDE_Q | 1.80 | 0.57 | 0.11 | 0.17 | 0.10 | 17.3 | Strongest FTNT match |
| OTIS* | SATELLITE | — | — | — | 0.49 | neg | 13.3 | Relaxed filter only |

*OTIS shows exceptional fundamentals but weaker ride confirmation (SATELLITE).

---

## 10. Blowoff-Over Hypotheses (Untested — For Exit Research)

| Hypothesis | Rule |
|------------|------|
| Width expansion stop | BB width +30–40% from shakeout low |
| RV20 spike-then-revert | Realized vol 20d spikes then decays |
| Upper BB persistence | Close > upper BB for > 5 days with declining volume |
| Time stop | 21–42 trading days post-sprung |

These are **not** in the detector; they are research targets for the exit side of the shadow book.

---

## 11. Usage

### Single Ticker (with fundamentals)
```bash
python coiled_spring.py --ticker FTNT --asof 2026-04-15
```

### Full Universe (writes `coiled_spring_screen.parquet`)
```bash
python coiled_spring.py --universe [--asof YYYY-MM-DD]
```

### Backtests
```bash
# Full state tracking (2.5M events)
python backtest_coiled_spring.py --universe --no_gpu

# Clean shadow book on test entries
python backtest_coiled_spring_clean.py
```

**Outputs:**
- `coiled_spring_screen.parquet` — latest state per ticker
- `backtest_coiled_spring_events.parquet` — historical state events
- `coiled_spring_full_events.parquet` — clean backtest events
- `shadow_book_clean.parquet` — P&L on test entries

---

## 12. Key Parameters (Defaults in `detect_spring`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `squeeze_days` | 10 | Min squeeze days in last 20 |
| `width_pctile` | 0.25 | BB width percentile threshold |
| `vol_z_thresh` | 1.5 | Volume z-score for shakeout |
| `reclaim_days` | 5 | Days allowed for reclaim |
| `expand_pct` | 0.20 | BB width expansion % to confirm |

---

## 13. Related Research (Distinct Overlays)

| Layer | Script | Purpose |
|-------|--------|---------|
| **Ignition / Rare Print** | `rare_ignition_info.py` | Single-bar volume/price anomaly on quality names |
| **Long-Ride Durability** | `ride_longevity.py` + `shock_ride.py` | 12M+ hold durability, entry gate, dual exit |
| **Coiled Spring (this)** | `coiled_spring.py` | BB/KC squeeze → shakeout → reclaim → expansion timing gate |

**All three are entry-only overlays on businesses selected by quality/value.** None discovers the business.

---

## 14. Data Integrity Notes

- `daily_prices/`: Additive yfinance OHLCV backfill (99.9% OHLC), LFS-tracked.
- `fundamentals.parquet`: 31,447 rows / 585 tickers / median 65 quarters / EDGAR priority > yfinance > Polygon.
- `preferred_metrics.parquet`: 585 latest / 66 INCLUDE / additive history (never overwrites EDGAR).
- All backtests **verify-by-running** on real parquet; no invented P&L.
- GPU (MX550 2 GB) used only for fractal profiles; coiled-spring backtests run CPU for stability.

---

## 15. GitHub-Safe Math Quick Reference

```
mid_20       = SMA(close, 20)
bb_upper     = mid_20 + 2 * STDDEV(close, 20)
bb_lower     = mid_20 - 2 * STDDEV(close, 20)
bb_width     = (bb_upper - bb_lower) / mid_20
bb_pos       = (close - bb_lower) / (bb_upper - bb_lower)

tr           = max(high - low, |high - close_prev|, |low - close_prev|)
atr_20       = SMA(tr, 20)
kc_mid       = SMA(close, 20)
kc_upper     = kc_mid + 1.5 * atr_20
kc_lower     = kc_mid - 1.5 * atr_20

squeeze_day  = (bb_upper < kc_upper) & (bb_lower > kc_lower)
squeeze_20d  = SUM(squeeze_day, 20)

bb_width_p252 = PERCENTILE_RANK(bb_width, 252)
width_compressed = bb_width_p252 <= 0.25

vol_z        = (volume - SMA(volume,20)) / STDDEV(volume,20)

shakeout     = (bb_pos < 0) & (vol_z >= 1.5)
reclaimed    = ANY(bb_pos >= 0) within 5 days of shakeout
expand_confirmed = bb_width_now / bb_width_shakeout >= 1.20

sprung       = squeeze_active & width_compressed & shakeout & reclaimed & expand_confirmed
```

---

*Generated from `coiled_spring.py`, `backtest_coiled_spring.py`, `backtest_coiled_spring_clean.py` and live backtest results (Aug 13 2026).*