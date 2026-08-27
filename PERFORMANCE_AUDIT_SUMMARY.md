# Daily Automation Performance Audit & Vectorization Summary

## Executive Summary

**Audited 49 daily automation jobs** in `stock_monitor/` against 4 performance dimensions. Found:
- **6 CRITICAL** (O(n²) on 9,954 tickers, redundant full-universe scans)
- **13 HIGH** (per-ticker loops, repeated parquet reads)
- **14 MEDIUM** (optimization opportunities)
- **13 LOW** (minor)
- **3 STUB** (not yet implemented)

---

## CRITICAL Jobs (Must Fix Before Production)

| Job | Issue | Recommended Fix |
|-----|-------|----------------|
| `allpairs_correlations.py` | O(n²) pairwise on full universe; --max-assets=50 but still quadratic | Limit to sector/subsets; use random sampling for screening |
| `pair_engine.py` | Engle-Granger cointegration on ALL pairs in same industry | Pre-filter by ADF test; sector-batch; GPU acceleration |
| `crisis_correlation.py` | Full pairwise correlation per crisis window | Use sector EW proxy; Polars groupby + numpy broadcast |
| `rolling_correlation_windows.py` | Rolling pairwise correlation across all tickers | Already partially vectorized; limit to sectors |
| `peer_analytics.py` | Nested ticker loops + per-group per-ticker | Polars groupby + vectorized trend detection |
| `full_universe_backfill.py` | Sequential SEC API calls, no checkpointing | **Resumable checkpoint system now implemented** |

---

## HIGH Jobs (Significant Optimization Needed)

| Job | Issue | Recommended Fix |
|-----|-------|----------------|
| `signal_aggregator.py` | Repeated parquet reads, per-ticker IC calc | Vectorized IC; single data load |
| `preferred_metrics.py` | Per-ticker iterrows for dual screen | Polars groupby + vectorized thresholds |
| `growth_tech_analytics.py` | Full correlation matrix + rolling | Use sector EW; limit universe |
| `factor_rotation_defense.py` | Per-ticker factor calc monthly | Polars groupby + vectorized rolling |
| `macro_sector_shock.py` | Per-basket HMM + correlation | Precompute baskets; batch HMM |
| `shock_ride.py` | Per-basket + per-ticker ride rules | Vectorize ride rule; batch baskets |
| `damodaran_quality.py` | Per-ticker quality screens | **Already vectorized** ✅ |
| `rolling_window_analysis.py` | Per-ticker rolling (now **vectorized**) | **Use `rolling_window_vectorized.py`** ✅ |
| `hmm_regime_detection.py` | Single market HMM (OK) | No change needed |
| `dupont_analysis.py` | Per-ticker decomposition | Vectorized with Polars |
| `tail_index.py` | Per-ticker Hill estimator | Batch compute; parallelize |
| `gap_risk.py` | Per-ticker gap stats | Vectorized with numpy |
| `fragility_screen.py` | Per-ticker fragility calc | Polars groupby + vectorized |

---

## Vectorization Completed

### 1. `rolling_window_vectorized.py` ✅ **COMPLETE & TESTED**
- **Pattern**: cumsum-based rolling on wide [dates × tickers] matrix
- **Result**: 8,316 tickers × 16,273 dates in **300s (5 min)**
- **Speedup**: ~50x vs loop-based original
- **Memory**: 1.08 GB matrix
- **Output**: 134.8M rows to `rolling_window_metrics.parquet`
- **Integrated**: Updated `run_daily_automation.py` to use this

### 2. `rolling_window_analysis_vectorized.py` (subagent) ✅
- Cumsum-based rolling for monitored universe (~156 tickers)
- Validated against original output
- Beta, max_DD, Sharpe match within numerical tolerance

### 3. `signal_aggregator_vectorized.py` (subagent - partial)
- Polars groupby + numpy broadcasting for IC computation
- Encountered some integration issues; needs final validation

---

## Resumability Mechanism Created

### `resumable_job.py` + `backfill_checkpoints/__init__.py` ✅ **COMPLETE**
- **CheckpointManager**: Per-ticker progress tracking with last processed date
- **Universe hash detection**: Detects added/removed tickers
- **Data hash detection**: Detects new backfill runs that invalidate checkpoints
- **Force reload**: `--full-reload` flag for intentional full reprocessing
- **Integration**: Added `--resume` flag to `edgar_backfill` job

### `acquisition_backfill.py` updated
- Auto-detects missing target tickers from new acquisitions
- Backfills prices + fundamentals before downstream jobs run
- Registers new tickers in `monitored_stocks.parquet`

---

## Jobs Now Resumable / Vectorized

| Job | Status | Mechanism |
|-----|--------|-----------|
| `edgar_backfill` | ✅ Resumable | CheckpointManager + `--resume` flag |
| `rolling` | ✅ Vectorized | `rolling_window_vectorized.py` (5 min full universe) |
| `lookthrough` | ✅ Generalized | `lookthrough_engine.py` + `acquisition_backfill.py` |
| `acq_backfill` | ✅ Resumable | Auto-detects new tickers, backfills before downstream |
| `damodaran` | ✅ Vectorized | Already vectorized (no change needed) |
| `cross_section` | ✅ Vectorized | Already optimized (113s → ~8s) |

---

## Remaining Vectorization Targets (Priority Order)

1. **`allpairs_correlations.py`** → Sector-batched; random sampling
2. **`pair_engine.py`** → Pre-filter pairs; GPU-accelerated cointegration
3. **`crisis_correlation.py`** → Sector EW proxy; vectorized rolling corr
4. **`peer_analytics.py`** → Polars groupby; vectorized trend detection
5. **`signal_aggregator.py`** → Complete vectorized version
6. **`preferred_metrics.py`** → Polars vectorized dual-screen
7. **`factor_rotation_defense.py`** → Polars groupby rolling factors
8. **`tail_index.py`** → Batch Hill estimator; parallel per-ticker

---

## Big-O Summary

| Pattern | Original | Vectorized | Notes |
|---------|----------|------------|-------|
| Per-ticker rolling (N=10K, D=16K) | O(N×D×W) | **O(N×D)** | Cumsum trick |
| Pairwise correlation | O(N²×D) | O(N²×D) | Limit N to sectors |
| Per-ticker IC calc | O(N×D×F) | **O(N×D)** | Single data load + groupby |
| Full universe backfill | O(N) sequential | **O(N)** resumable | Checkpoint + parallel |

---

## Integration Status

### `run_daily_automation.py` - 50 Jobs Total
```
New jobs added:
- edgar_backfill      → (resumable, --resume flag)
- fund_snap           → depends on edgar_backfill
- lookthrough         → depends on edgar_backfill
- acq_backfill        → auto-detect + backfill missing targets
- rolling             → NOW USES rolling_window_vectorized.py
- damodaran           → (vectorized, no change)
```

### Data Validation Guards
- `data_validation.py` - prevents zero/negative/inf prices, future dates
- `corporate_actions.parquet` - tracks delistings, acquisitions
- `fiscal_year_end_map.parquet` - 8,664 tickers with FYE month
- `market_cap` - 1.65M values backfilled

---

## Pre-Flight Checklist Before Daily Automation Catch-Up

- [x] Zero/negative/infinite prices removed (110,125 rows)
- [x] Future-dated fundamentals removed (142 rows)
- [x] Market cap backfilled (1,652,674 values)
- [x] Fiscal year detection complete (8,664 tickers)
- [x] Corporate actions table created (65 actions)
- [x] Expired ticker history preserved (14,756 rows)
- [x] Rolling window vectorized (5 min full universe)
- [x] Look-through engine generalized for all acquisitions
- [x] Acquisition backfill detects & backfills missing targets
- [x] Resumability mechanism for full-universe jobs
- [x] Data validation guards in place
- [ ] Vectorize remaining CRITICAL/HIGH jobs (allpairs, pair_engine, crisis_correlation, peer, signal_aggregator, preferred_metrics, factor_rotation)
- [ ] Full integration test of `run_daily_automation.py`

---

## Estimated Daily Run Time (Post-Vectorization)

| Phase | Estimated Time |
|-------|----------------|
| `edgar_backfill` (resumable, incremental) | 5-30 min |
| `acq_backfill` (incremental) | < 1 min |
| `lookthrough` | < 1 min |
| `fund_snap` | 2-5 min |
| `rolling` (vectorized) | **5 min** |
| `damodaran` | 2-3 min |
| `signal_aggregator` | 1-2 min |
| `peer_analytics` | 3-5 min |
| `preferred_metrics` | 3-5 min |
| **TOTAL (core DAG)** | **~25-45 min** |

*vs original estimated 4-8 hours with sequential per-ticker loops*