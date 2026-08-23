# Quality Gate vs Novy-Marx — Phase 1.1 final

**Closed:** 2026-08-23  
**Verdict:** Gate ∩ NM-quality = **60/95 = 63%** after persisted component D/E (bar 80% — **fail**). Pre-persist dated-asset panel was 49/82 = 60%. `buffett_pass` is Buffett ROE/ROIC/D/E. It is **not** QMJ. **Do not loosen 15/15/1.0.**

**Inputs:** `preferred_metrics.parquet` (8,669), `fundamentals.parquet` (323,150 dated rows), `novymarx_*.parquet` (3,665 filing dates), `quality_gate_comparison.parquet`

---

## What we measured

Live gate (`analytics_common.BASE_THRESHOLDS`): **ROE ≥ 15%, ROIC ≥ 15%, D/E ≤ 1.0**. `buffett_pass` = all three.

NM-quality := mean rank ≥ 0.5 on **≥2** of {high Rev/A, low filing AG, low accruals, low fund D/E}. AG clipped ±100% for ranks only.

| Rule | Count / 8,669 |
|------|----------------|
| ROE ≥ 15% | 306 (3.5%) |
| ROIC ≥ 15% | 187 (2.2%) |
| D/E ≤ 1.0 (recomputed) | 3,868 |
| **buffett_pass** | **98** |
| INCLUDE_CORE | **22** |
| INCLUDE_VALUE | 8 |

NM panels (`factor_library.py --quality-only`, filing calendar):

| Panel | Tickers | Last non-null |
|-------|---------|---------------|
| asset_growth | **8,983** | 8,983 |
| gross_profitability (Rev/A) | 9,121 | 6,281 |
| accruals | 9,123 | 7,031 |
| debt_to_equity | 9,005 | 6,101 |
| book_to_market | 8,924 | 5,499 |

| NM leg | Universe median | Gate median | Gate %ile | Read |
|--------|-----------------|-------------|-----------|------|
| Rev/A | 0.369 | **0.682** | **68** | quality |
| Filing AG | +0.77% | **0.00%** | **39** | slightly conservative |
| Accruals | −0.036 | −0.022 | **58** | worse |
| Fund D/E | 0.289 | 0.341 | **53** | none |
| B/M | 0.388 | 0.132 | **29** | **growth, not value** |

Universe NM-quality: **3,796 / 8,126** (46.7%).

| Set | Eligible (≥2 NM legs) | NM-quality | Overlap |
|-----|----------------------|------------|---------|
| buffett_pass (98) | 95 | 60 | **63%** |

Lowest NM scores among the 85: SCHW, SAGT (AG +705%), MLM, GOOG/GOOGL, MSI, HUBB, APH, GL, JBL, SMG, AEM.

---

## What Phase 1.1 shipped

| Item | State |
|------|--------|
| Dated `total_assets` timeseries | 323,150 rows; 8,992 names with ≥2 dates |
| Filing-to-filing AG | `_asset_growth_from_filings` |
| NM panels | 3,665 × ~9k, not the 111-name stub |
| `nm_quality` / `nm_score` / `nm_legs` | `factor_library.attach_nm_quality` |
| Component D/E | `total_debt / shareholders_equity` when book equity > 0 |
| `buffett_leverage` | requires `D/E ≥ 0` |
| `gross_profit` in edgar v2 | AAPL TTM GP present; NM still Rev/A until panel backfilled |
| FF5 `--full` column map | `shareholders_equity` / `revenue_ttm`\|`gross_profit` / filing AG |
| `ff5_factors.parquet` | MKT/SMB/HML/CMA/MOM written. **MKT −6.4% / 16% vol**, corr TMI **0.78**. RMW empty (no GP). |
| Residual IC ≥ +0.02 | **+0.0117** / 85m CAPM residual on **fixed MKT** (bar fail) |

D/E recompute **persisted** 2026-08-23: D/E n=5,581 (was 805). Drops negative-D/E names. Adds BIIB, GOLD, JPM, RJF, SLB to CORE. Drops SBUX. **INCLUDE_CORE = 22.**

---

## Corrective work (do not loosen 15/15/1.0)

**Done**
- `nm_quality` persisted on `preferred_metrics.parquet`
- QMI = top quintile of `nm_score` (≥2 legs): **1,531** names
- Dual-pass `value_pass`. **INCLUDE_CORE = 22**: AEM, AGI, ALL, BEN, BIIB, CAG, CPRT, CTAS, EOG, GL, GOLD, HAL, HBM, JPM, PHM, RJF, SCHW, SLB, SYF, TSM, UNH, WMT

**Open**
- Backfill `gross_profit` then GP/A (RMW is 0 without it)
- Residual IC still below +0.02 after MKT fix

Rebuild NM: `python factor_library.py --quality-only --save`
