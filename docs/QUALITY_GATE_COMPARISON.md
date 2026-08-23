# Quality Gate vs Novy-Marx

**As of:** 2026-08-23  
**Inputs:** `preferred_metrics.parquet` (8,669), `novymarx_*.parquet` (filing calendar, 3,665 dates), `quality_gate_comparison.parquet`

Live quality gate (`analytics_common.BASE_THRESHOLDS`): **ROE ≥ 15%, ROIC ≥ 15%, D/E ≤ 1.0**. `buffett_pass` is all three.

Novy-Marx legs (last non-null filing per ticker):

| Leg | Definition | Quality direction |
|-----|------------|-------------------|
| Profitability | `revenue_ttm / total_assets` (Rev/A; no COGS) | high |
| Investment | consecutive-filing `%Δ total_assets` | low |
| Accruals | `(NI_ttm − OCF_ttm) / total_assets` | low |
| Leverage | `total_debt / shareholders_equity` | low |
| Book/Market | `equity / market_cap` | high if value |

NM-quality := mean of available quality ranks ≥ 0.5 on **≥2** of {GP, low AG, low accruals, low D/E}.

---

## Our gate

| Rule | Count / 8,669 |
|------|----------------|
| ROE ≥ 15% | 306 (3.5%) |
| ROIC ≥ 15% | 187 (2.2%) |
| D/E ≤ 1.0 | 565 (6.5%) |
| ≥2 of 3 legs | 246 (2.8%) |
| **buffett_pass (all 3)** | **85 (1.0%)** |
| INCLUDE_QUALITY | 21 |
| INCLUDE_CORE | **0** |
| INCLUDE_VALUE (trifecta) | 8 |

**buffett_pass (85):**  
AAPL, ACN, ADBE, ADP, AEM, AGI, ALL, AMAT, AOS, APH, ASML, BBY, BEN, BLDR, CAG, CF, CHAI, CHKP, CMI, COST, CPRT, CTAS, DGX, DLTR, DRI, DXCM, EME, EOG, EW, EXP, FAST, FFIV, FIX, GE, GL, GOOG, GOOGL, GWW, HAL, HBM, HERE, HUBB, IDXX, JBL, JKHY, JNJ, KGC, LRCX, MA, MLM, MNST, MSFT, MSI, NVDA, NVR, ODFL, OTIS, PAYX, PG, PHM, PPG, PYPL, RMD, ROK, ROL, ROST, SAGT, SBUX, SCCO, SCHW, SMG, SNA, SYF, TDG, TER, TMUS, TSCO, TSM, TT, UNH, VIG, VLO, VRTX, WMT, WST

**INCLUDE_QUALITY (21):**  
ADBE, AEM, AGI, ALL, BBY, BEN, CAG, CF, EOG, GL, HBM, KGC, MSFT, PYPL, SBUX, SMG, SYF, TDG, TSM, UNH, VLO

---

## Novy-Marx panels (dated `total_assets`, ≥2 filings)

`factor_library.py --quality-only` on the filing calendar (3,665 dates, not a 1967–2026 daily grid).

| Panel | Tickers | Latest non-null |
|-------|---------|-----------------|
| `novymarx_asset_growth` | **8,983** | 8,983 |
| `novymarx_gross_profitability` | 9,121 | 6,281 |
| `novymarx_accruals` | 9,123 | 7,031 |
| `novymarx_debt_to_equity` | 9,005 | 6,101 |
| `novymarx_book_to_market` | 8,924 | 5,499 |

| Metric | n | All median | buffett_pass n | Gate median | Gate %ile | vs NM quality |
|--------|---|------------|----------------|-------------|-----------|---------------|
| Gross profitability (Rev/A) | 6,281 | 0.369 | 81 | **0.682** | **68** | more profitable |
| Asset growth (filing %Δ) | 8,983 | +0.77% | 84 | **0.00%** | **39** | slightly conservative |
| Accruals (NI−OCF)/A | 7,031 | −0.036 | 79 | −0.022 | **58** | *higher* accruals |
| Debt/Equity (fund) | 6,101 | 0.289 | 81 | 0.341 | **53** | no leverage tilt |
| Book/Market | 5,499 | 0.388 | 79 | 0.132 | **29** | **growth, not value** |

Universe NM-quality: **3,796 / 8,126** (46.7%).

---

## Overlap (plan bar: ≥80%)

| Set | Eligible (≥2 NM legs) | NM-quality | Overlap |
|-----|----------------------|------------|---------|
| buffett_pass (85) | 82 | 49 | **60%** |
| INCLUDE_QUALITY (21) | 21 | 10 | **48%** |

**Verdict:** on the expanded ≥2-date panel the gate is **high Rev/A, slightly low investment**. Accruals and fund D/E are not quality-tilted. It is **not** a value screen. Overlap **60% < 80%** — Gate 1 quality-overlap metric **fails**.

33 `buffett_pass` names miss NM-quality. Lowest scores: SCHW, SAGT (AG +705%), MLM, GOOG/GOOGL, MSI, HUBB, APH, GL, JBL, SMG, AEM. Typical miss: high asset growth and/or high fund D/E.

QMI’s 8-name book is the same tightness: `buffett_pass` is 1% of names; INCLUDE_CORE is 0.

---

## What the 85-name list is

A Buffett ROE/ROIC/D/E cut on `preferred_metrics`. It is **not** Novy-Marx QMJ. Rev/A is the GP proxy until COGS exists.

---

## Follow-up (do not loosen 15/15/1.0)

1. Separate `nm_quality` flag; do not reuse `buffett_pass` as QMJ.
2. Reconcile preferred D/E vs fund D/E (gate names sit at the 53rd %ile of fund D/E).
3. INCLUDE_CORE = 0 — dual-pass needs B/M or EY, not the current trifecta.
4. QMI: rank QMJ instead of hard-threshold 15/15/1.0.
5. True GP/A needs COGS.
6. Winsorize AG before ranking (SAGT +705%).
7. Wire HML/RMW/CMA from equity / Rev/A / filing AG.
8. Re-run residual IC only after (7).

---

## Artifacts

- `novymarx_{gross_profitability,asset_growth,accruals,debt_to_equity,book_to_market}.parquet` — filing-date panels
- `quality_gate_comparison.parquet` — one row/ticker
- Rebuild: `python factor_library.py --quality-only --save`
