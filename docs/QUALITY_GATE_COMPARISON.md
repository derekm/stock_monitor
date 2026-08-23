# Quality Gate vs Novy-Marx

**As of:** 2026-08-23  
**Inputs:** `preferred_metrics.parquet` (8,669 tickers), `fundamentals.parquet` (323,150 dated rows), `quality_gate_comparison.parquet`

Live quality gate (`analytics_common.BASE_THRESHOLDS`): **ROE ≥ 15%, ROIC ≥ 15%, D/E ≤ 1.0**. `buffett_pass` is all three.

Novy-Marx legs (latest filing per ticker):

| Leg | Definition | Quality direction |
|-----|------------|-------------------|
| Profitability | `revenue_ttm / total_assets` (Rev/A; no COGS panel) | high |
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

## Novy-Marx latest cross-section (dated `total_assets`)

8,992 names have ≥2 asset observations (median 23 dates). Latest-filing coverage:

| Metric | Latest n | All median | buffett_pass n | Gate median | Gate %ile | vs NM quality |
|--------|----------|------------|----------------|-------------|-----------|---------------|
| Gross profitability (Rev/A) | 4,837 | 0.358 | 43 | **0.825** | **75** | more profitable |
| Asset growth (filing %Δ) | 8,983 | +0.77% | 84 | **0.00%** | **39** | slightly conservative |
| Accruals (NI−OCF)/A | 5,556 | −0.033 | 42 | −0.038 | **47** | neutral |
| Debt/Equity (fund) | 5,242 | 0.320 | 46 | 0.333 | **51** | no leverage tilt |
| Book/Market | 4,509 | 0.406 | 42 | 0.115 | **26** | **growth, not value** |
| Book/Market (1 / P/B) | 521 | 0.287 | 72 | 0.170 | **34** | not cheap |

Universe NM-quality: **3,293 / 7,081** names with ≥2 legs (46.5%).

---

## Overlap (plan bar: ≥80%)

| Set | Eligible (≥2 NM legs) | NM-quality | Overlap |
|-----|----------------------|------------|---------|
| buffett_pass (85) | 46 | 31 | **67%** |
| INCLUDE_QUALITY (21) | 3 | 2 | **67%** (n=3) |

**Verdict:** with dated assets, the gate is a **high-Rev/A, slightly low-investment** cut. Accruals and fund D/E are universe-median. It is **not** a value screen. Overlap **67% < 80%** — Gate 1 quality-overlap metric **fails**.

15 `buffett_pass` names with ≥2 NM legs miss NM-quality. Lowest NM scores: MLM, GOOG, MSI, SAGT (AG +705%), APH, GL, JBL, GE, MA, EXP. Typical miss: high asset growth and/or high fund D/E despite passing preferred D/E.

QMI’s 8-name book is the same tightness: `buffett_pass` is 1% of names; INCLUDE_CORE is 0.

---

## What the 85-name list is

A Buffett ROE/ROIC/D/E cut on `preferred_metrics`. It is **not** Novy-Marx QMJ (GP/A + low investment + low accruals). Rev/A is the GP proxy until COGS exists.

---

## Follow-up: gate corrections (do not ship without a remeasure)

Do **not** loosen ROE/ROIC/D/E to chase the 80% bar. That would inflate `buffett_pass` without adding QMJ content.

1. **Split the screens.** Keep `buffett_pass` as the ROE/ROIC/D/E cut. Add a separate `nm_quality` flag (rank ≥0.5 on ≥2 of GP / low AG / low accruals / low fund D/E) and require it for QMI / INCLUDE_QUALITY if the goal is QMJ.
2. **Reconcile D/E.** Preferred-gate D/E ≤ 1.0 but latest fund D/E for the 85 is universe-median (51st %ile). Same name, two series — pick one for the gate.
3. **INCLUDE_CORE = 0.** Dual-pass (quality ∧ value trifecta) never fires at these thresholds. Either the trifecta is the wrong value definition, or dual-pass should use NM B/M / EY instead of EV/EBITDA + P/B + MCA.
4. **QMI 8 names.** Same root: the quality gate is 1% of the universe. A QMJ sleeve should rank, not hard-threshold 15/15/1.0.
5. **True GP/A.** Rev/A overstates profitability for high-COGS names. Need `gross_profit` / COGS from companyfacts before treating GP as validated.
6. **Winsorize AG.** SAGT +705% is a restatement/unit change, not investment. Cap AG at, say, ±100% before ranking.
7. **FF5 HML/RMW.** Still empty in `ff5_factors.parquet` (price-only MKT/SMB/MOM). Wire `shareholders_equity` / Rev/A / filing AG into `compute_ff5_with_fundamentals` before factor-adjusting signals again.
8. **IC ≥ +0.02 after residual.** Not re-measured on the dated panel. Re-run `signal_aggregator.py --use-residuals` only after HML/RMW exist.

---

## Artifacts

- `quality_gate_comparison.parquet` — one row/ticker: gate flags + latest NM columns
- `fundamentals.parquet` — dated `total_assets` (323,150 rows)
- Rebuild comparison: latest-filing join of `fundamentals` → consecutive-filing `%Δ assets` → rank vs `preferred_metrics`
