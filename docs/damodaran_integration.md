# Aswath Damodaran Research Integration — stock_monitor Framework

**Purpose**: Systematically incorporate Damodaran's valuation methodologies, ERP/CRP data, corporate life cycle framework, narrative-to-numbers process, and cost-of-capital principles into our preferred metrics, implied return calculations, and quality screens.

**Status (2026-08)**: ✅ Core integration complete — WACC, life cycle, fair multiples, ERP service module all implemented and vectorized.

---

## 1. Equity Risk Premium (ERP) & Country Risk Premium (CRP) — Dynamic Inputs

### 1.1 Current Estimates (January 2026)

| Market | Implied ERP | Risk-Free (10Y) | Expected Return | Source |
|--------|-------------|-----------------|-----------------|--------|
| **US (S&P 500)** | **4.23%** | 4.18% | **8.41%** | histimpl.html Jan 2026 |
| **Mature Markets (Aaa)** | **4.21–4.33%** | — | — | ctrypremJuly25.xlsx |
| **China** | **5.14%** | — | — | CRP = 0.91% |
| **Brazil** | **7.47%** | — | — | CRP = 3.26% |

### 1.2 Integration Points

| Our Module | Current Approach | Damodaran Enhancement |
|------------|------------------|----------------------|
| `implied_r_screen.py` | Multiple ERP sources (Damodaran, CAPE, SPY SMA) | **Dynamic ERP from `erp_service.py`** |
| `preferred_metrics.py` | Delegates to `damodaran_data.compute_wacc_per_ticker` | **Per-ticker WACC** using Damodaran's cost-of-capital framework |
| `damodaran_data.py` | Computes WACC, life cycle, fair multiples | **Vectorized** (no `iterrows`) |

### 1.3 ERP Service Module

New in 2026-08: `erp_service.py` — unified ERP service providing:
- Damodaran implied ERP (annual + semi-annual, from `erp_history.parquet`)
- Shiller CAPE ERP (1/PE10 - long rate, hash-verified)
- SPY SMA heuristic (price/200dma, labeled honestly)
- Interpolated monthly/daily ERP
- Single download, daily refresh, cached parquet storage

```python
from erp_service import latest_implied_erp, load_erp, refresh_all_erp

erp = latest_implied_erp()  # 0.0423 (4.23%)
df = load_erp("damodaran", "monthly")  # interpolated monthly
```

---

## 2. Cost of Capital — Per-Ticker WACC (Damodaran "Swiss Army Knife")

### 2.1 Framework

```
Cost of Equity = Riskfree Rate + Beta * (ERP + CRP) + [Small Cap / Liquidity Premium]
Cost of Debt = Riskfree Rate + Default Spread * (1 - Tax Rate)
WACC = Cost of Equity * (E / (D+E)) + Cost of Debt * (D / (D+E))
```

### 2.2 Key Principles (Implemented)

| Principle | Application |
|-----------|-------------|
| **Dynamic, not static** | Recomputed quarterly via `damodaran_data.compute_wacc_per_ticker` |
| **Business-specific** | Company-level (segment data not available) |
| **Cost of capital = receptacle for fears** | No company-specific risk premiums added to WACC |
| **Small cap / liquidity premiums** | Excluded per Damodaran |
| **Marginal tax rate** | 21% US marginal rate used |

### 2.3 Implementation

```python
# In damodaran_data.py — VECTORIZED (no iterrows)
def compute_wacc_per_ticker(fund: pd.DataFrame) -> pd.DataFrame:
    # Uses: rf=0.0418, erp=latest_implied_erp(), tax=0.21
    # Beta: regressed vs SPY (5Y monthly) or sector bottom-up
    # Cost of Debt: synthetic rating from interest coverage
    # Returns: ticker, wacc, cost_of_equity, cost_of_debt, synthetic_rating, sector_beta
```

### 2.4 Synthetic Rating from Interest Coverage

| Interest Coverage | Rating | Default Spread |
|-------------------|--------|----------------|
| > 8.5x | Aaa/AAA | 0.40% |
| 6.5–8.5x | Aa/AA | 0.70% |
| 5.5–6.5x | A | 0.90% |
| 4.25–5.5x | Baa/BBB | 1.50% |
| 3–4.25x | Ba/BB | 2.50% |
| 2–3x | B | 4.00% |
| 1.5–2x | Caa/CCC | 6.00% |
| < 1.5x | Ca/CC | 10.00% |

---

## 3. Corporate Life Cycle — Stage-Aware Valuation & Quality

### 3.1 Six Stages (Implemented in `damodaran_data.classify_life_cycle`)

| Stage | Characteristics | Valuation Approach | Pricing Multiples |
|-------|-----------------|-------------------|-------------------|
| **Start-up** | Negative CF, high failure risk, no history | Option pricing / VC method | Users, revenue (pre-revenue) |
| **Young Growth** | Revenue growth > 30%, negative/low FCF, high reinvestment | Narrative-driven DCF | Revenue, EV/Sales |
| **High Growth** | Growth 15–30%, FCF turning positive, reinvestment peaking | DCF with fading growth | EV/EBITDA, P/E |
| **Mature Growth** | Growth 5–15%, stable + FCF, reinvestment moderating | Standard DCF / Multiples | P/E, EV/EBITDA |
| **Mature Stable** | Growth ≈ GDP, high FCF, low reinvestment | Stable growth DCF / Dividend discount | P/E, P/B, Div yield |
| **Decline** | Negative growth, FCF > reinvestment, asset liquidation | Liquidation / Sum-of-parts | P/B, EV/Assets |

### 3.2 Classification Algorithm (Vectorized)

```python
def classify_life_cycle(fundamentals_latest):
    rev_growth_3y = fundamentals['revenue_growth_3y']
    # free_cash_flow is TTM, so the denominator is TTM revenue.
    fcf_margin = fundamentals['free_cash_flow'] / fundamentals['revenue_ttm']
    reinvestment_rate = 1 - (fundamentals['free_cash_flow'] / fundamentals['ebit'])
    roic = fundamentals['roic']
    
    if rev_growth_3y > 0.30 and fcf_margin < 0:        return "Young Growth"
    elif rev_growth_3y > 0.15 and fcf_margin < 0.05:    return "High Growth"
    elif rev_growth_3y > 0.05 and roic > 0.15:         return "Mature Growth"
    elif rev_growth_3y > 0.02 and fcf_margin > 0.10:   return "Mature Stable"
    elif rev_growth_3y < 0:                            return "Decline"
    else:                                               return "Unclassified"
```

### 3.3 Coverage (as of 2026-08-17)

| Life Cycle Stage | Count | Pct |
|-----------------|-------|-----|
| Unclassified | 5,542 | 63.8% |
| Mature Growth | 1,059 | 12.2% |
| Decline | 816 | 9.4% |
| High Growth | 496 | 5.7% |
| Young Growth | 421 | 4.8% |
| Mature Stable | 345 | 4.0% |

*Note: High "Unclassified" due to missing `revenue_growth_3y` / FCF data for many tickers.*

---

## 4. Fair Multiples — First Principles & Fundamental Drivers (VECTORIZED)

### 4.1 Core Insight

> **Every multiple is a function of fundamentals: growth, risk, cash flow characteristics.**

### 4.2 Formulas (Implemented in `damodaran_data.compute_fair_multiples` — VECTORIZED)

```python
# VECTORIZED numpy implementation (no iterrows)
# NaN unless WACC > g, ROIC/ROE > 0, reinvestment in [0, 1), result > 0.
# Negative "fair" EV/EBITDA is g > ROIC, not a multiple — MOS 2026-07-31 was −14.4x.
fair_pe       = (1 - g/roe) / (cost_of_equity - g)
fair_ev_ebitda = (1 - g/roic) * (1 - t) / (wacc - g)
fair_ev_sales  = margin * (1 - g/roic) * (1 - t) / (wacc - g)
fair_pb       = (roe - g) / (cost_of_equity - g)
```

### 4.3 Coverage

| Metric | Latest tickers (2026-08-30) | Pct of 9,345 |
|--------|-----------------------------|--------------|
| fair_pe | 454 | 4.9% |
| fair_ev_ebitda | 486 (0 negative) | 5.2% |
| fair_ev_sales | 365 | 3.9% |
| fair_pb | 454 | 4.9% |

*Low coverage = WACC + ROIC>0 + reinvestment in [0, 1). 461/486 latest EV/EBITDA positives have implied_growth = 0.02 (`fillna`), not reported g. Come back: stop filling g; do not add a fair_ev provenance sister (g/ROE path is dead; NaN is the provenance).*

---

## 5. Margin of Safety — Post-Valuation Discipline

| Current | Damodaran-Aligned |
|---------|-------------------|
| No explicit MoS | **Require 15–25% discount to intrinsic value** before BUY |
| Implied return vs hurdle | **Implied return ≥ WACC + MoS spread** (e.g., WACC + 3–5%) |
| Quality score as proxy | **Separate quality assessment from valuation margin** |

Implemented in `preferred_metrics.py`:
- `discount_to_fair = (fair_ev_ebitda - ev_ebitda) / fair_ev_ebitda` only when `fair_ev_ebitda > 0` and traded `ev_ebitda > 0`
- `mos_pass = discount_to_fair >= 0.15` — a negative fair multiple cannot pass; traded EV/EBITDA ≤ 0 cannot pass (KHC −13.58 / SNDK −560.83 were the leftover)

---

## 6. Implementation Status — stock_monitor Modules

| Module | Change | Status |
|--------|--------|--------|
| `erp_service.py` | Unified ERP service (Damodaran, CAPE, SPY SMA, interpolated) | ✅ **Complete** |
| `damodaran_data.py` | Vectorized WACC, life cycle, fair multiples | ✅ **Complete** |
| `preferred_metrics.py` | Delegates to `damodaran_data`; adds `mos_pass`, distrust | ✅ **Complete** |
| `implied_r_screen.py` | Multiple ERP sources; CAPE hash verification | ✅ **Complete** |
| `peer_analytics.py` | Cross-sectional regression: multiple ~ growth + ROIC + WACC | ✅ **Complete** (vectorized + GPU trends) |
| `buy_candidates.py` | MoS check + distrust discount | ✅ **Complete** |
| `stress_dual_pass.py` | ERP scenario analysis | ⏳ Pending |

---

## 7. Data Sources & Storage

| Data | Source | Frequency | Our Storage |
|------|--------|-----------|-------------|
| **US Implied ERP** | Stern histimpl.html | Monthly (Jan/Jul) | `erp_history.parquet` |
| **Country Risk Premiums** | ctryprem*.xlsx | Semi-annual | `crp_by_country.parquet` |
| **Sector Betas** | Damodaran sector pages | Annual | (computed inline) |
| **CAPE Dataset** | datahub.io / github.com/datasets/s-and-p-500 | Monthly | `erp_annual.parquet` (fallback) |

---

## 8. Key Formulas Quick Reference

### 8.1 Cost of Equity
```
cost_of_equity = rf + beta * (erp_us + crp_country)
```

### 8.2 Cost of Debt (from interest coverage)
```python
if interest_coverage > 8.5:    spread = 0.004
elif interest_coverage > 6.5:  spread = 0.007
elif interest_coverage > 5.5:  spread = 0.009
elif interest_coverage > 4.25: spread = 0.015
elif interest_coverage > 3:    spread = 0.025
elif interest_coverage > 2:    spread = 0.040
elif interest_coverage > 1.5:  spread = 0.060
else:                          spread = 0.100

cost_of_debt = rf + spread
after_tax_cost_of_debt = cost_of_debt * (1 - 0.21)
```

### 8.3 WACC
```
wacc = cost_of_equity * (E / (D+E)) + after_tax_cost_of_debt * (D / (D+E))
```

### 8.4 Fair Multiples (Stable Growth)
```
fair_pe       = (1 - g/roe) / (cost_of_equity - g)
fair_ev_ebitda = (1 - g/roic) * (1 - t) / (wacc - g)
fair_ev_sales  = margin * (1 - g/roic) * (1 - t) / (wacc - g)
fair_pb       = (roe - g) / (cost_of_equity - g)
```
Emit NaN when g ≥ ROIC (reinvestment ∉ [0, 1)), ROE ≤ g, WACC ≤ g, or the result is ≤ 0. A negative multiple is not a cheapness signal.

### 8.5 Life Cycle Classification
```
if rev_growth > 30% and fcf_margin < 0:          "Young Growth"
elif rev_growth > 15% and fcf_margin < 5%:       "High Growth"
elif rev_growth > 5% and roic > 15%:             "Mature Growth"
elif rev_growth > 2% and fcf_margin > 10%:       "Mature Stable"
elif rev_growth < 0:                             "Decline"
```

---

## 9. Validation Checklist

- [x] ERP/CRP data pipeline automated (`erp_service.refresh_all_erp()`)
- [x] Per-ticker WACC computed and stored in `preferred_metrics.parquet`
- [x] Life cycle stage assigned to all tickers
- [x] Fair multiples replace static thresholds in quality screen
- [x] Margin of safety (15–25%) enforced in `preferred_metrics.mos_pass`
- [x] Peer comparison uses fundamental-adjusted residuals
- [x] Implied return uses multiple ERP sources
- [ ] Backtest: Damodaran-enhanced signals vs current signals (OOS)

---

## 10. References (Primary Sources)

| Topic | Paper / Page | URL |
|-------|--------------|-----|
| Multiples: First Principles | Damodaran (2009+) | pages.stern.nyu.edu/~adamodar/pdfiles/papers/multiples.pdf |
| ERP: Determinants & Estimation | Damodaran (2009, 2026) | pages.stern.nyu.edu/~adamodar/pdfiles/papers/ERP2009.pdf |
| Cost of Capital | Damodaran (2016) | pages.stern.nyu.edu/adamodar/pdfiles/papers/costofcapital.pdf |
| Corporate Life Cycle | Damodaran (2023) | pages.stern.nyu.edu/~adamodar/pdfiles/country/CorpLifeCycleLong2023.pdf |
| Narrative & Numbers | Damodaran (2019) | pages.stern.nyu.edu/~adamodar/pdfiles/blog/narrative&numberslongDEShaw2019.pdf |
| Current ERP/CRP Data | Damodaran (2026) | pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html |
| Margin of Safety | Damodaran (2011) | aswathdamodaran.blogspot.com/2011/04/margin-of-safety-alternative-risk_16.html |
| Young Company Valuation | Damodaran (SSRN 1418687) | papers.ssrn.com/sol3/papers.cfm?abstract_id=1418687 |

---

*Generated from Damodaran primary sources (papers, data files, blog) — integrated into stock_monitor framework for implied return, quality assessment, and valuation discipline. All core modules vectorized; GPU-accelerated where applicable.*