# Quality Gate vs Novy-Marx

**As of:** 2026-08-23  
**Inputs:** `preferred_metrics.parquet` (8,669 tickers), `factor_library.py --quality-only` panels, `quality_gate_comparison.parquet`

Our live quality gate is `analytics_common.BASE_THRESHOLDS`: **ROE ≥ 15%, ROIC ≥ 15%, D/E ≤ 1.0**. `buffett_pass` is all three. Novy-Marx panels: profitability = `revenue_ttm / total_assets` (no COGS), investment = consecutive-filing `%Δ total_assets`, accruals = `(NI_ttm − OCF_ttm) / assets`, leverage = `total_debt / equity`, B/M = `equity / fundamentals.market_cap`.

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
| INCLUDE_CORE | 0 |
| INCLUDE_VALUE (trifecta) | 8 |

**buffett_pass (85):**  
AAPL, ACN, ADBE, ADP, AEM, AGI, ALL, AMAT, AOS, APH, ASML, BBY, BEN, BLDR, CAG, CF, CHAI, CHKP, CMI, COST, CPRT, CTAS, DGX, DLTR, DRI, DXCM, EME, EOG, EW, EXP, FAST, FFIV, FIX, GE, GL, GOOG, GOOGL, GWW, HAL, HBM, HERE, HUBB, IDXX, JBL, JKHY, JNJ, KGC, LRCX, MA, MLM, MNST, MSFT, MSI, NVDA, NVR, ODFL, OTIS, PAYX, PG, PHM, PPG, PYPL, RMD, ROK, ROL, ROST, SAGT, SBUX, SCCO, SCHW, SMG, SNA, SYF, TDG, TER, TMUS, TSCO, TSM, TT, UNH, VIG, VLO, VRTX, WMT, WST

**INCLUDE_QUALITY (21):**  
ADBE, AEM, AGI, ALL, BBY, BEN, CAG, CF, EOG, GL, HBM, KGC, MSFT, PYPL, SBUX, SMG, SYF, TDG, TSM, UNH, VLO

---

## Novy-Marx latest cross-section

Asset growth exists only where `total_assets` has **≥2 filing dates**. That is 111 tickers (113 consecutive pairs). The fundamentals panel is ~1 row/ticker (mean 1.07), so investment coverage is structural, not a resampling choice.

| Metric | Latest n | All median | buffett_pass n | Gate median | Gate %ile | Direction vs NM quality |
|--------|----------|------------|----------------|-------------|-----------|-------------------------|
| Gross profitability (Rev/A) | 113 | 0.438 | 19 | **0.620** | **64** | higher profitability |
| Asset growth (filing %Δ) | 111 | −0.0025 | 18 | −0.0024 | **50** | no investment tilt |
| Accruals (NI−OCF)/A | 127 | −0.024 | 21 | −0.004 | **64** | *higher* accruals |
| Debt/Equity | 366 | 0.272 | 7 | **0.076** | **32** | lower leverage |
| Book/Market (fund mcap) | 176 | 0.200 | 22 | 0.142 | **43** | not cheap |
| Book/Market (1 / P/B) | 521 | 0.287 | 72 | 0.170 | **34** | not cheap |

---

## Overlap (plan bar: ≥80%)

NM-quality := mean of available quality ranks ≥ 0.5 on ≥2 of {high GP, low asset growth, low accruals, low D/E}.

| Set | Eligible (≥2 NM legs) | NM-quality | Overlap |
|-----|----------------------|------------|---------|
| buffett_pass (85) | 21 | 10 | **48%** |
| INCLUDE_QUALITY (21) | 12 | 6 | **50%** |

**Verdict:** the gate matches Novy-Marx on **profitability and leverage**. It does **not** match on investment or accruals, and it is **not** a value screen. Overlap **48% < 80%** — plan Gate 1 quality-overlap metric **fails**.

QMI’s 8-name universe is the same tightness: `buffett_pass` is 1% of names; INCLUDE_CORE is 0.

---

## What the 85-name list is

A high-ROE / high-ROIC / low-D/E cut. That is a Buffett-style quality screen, not a Novy-Marx QMJ replica. QMJ needs GP/A, low investment, and low accruals with enough history to rank them. We have those series for ~110–370 names.

---

## Artifacts

- `novymarx_{gross_profitability,asset_growth,accruals,debt_to_equity,book_to_market}.parquet`
- `quality_gate_comparison.parquet` — one row/ticker, gate flags + latest NM columns
- Rebuild: `python factor_library.py --quality-only --save`
