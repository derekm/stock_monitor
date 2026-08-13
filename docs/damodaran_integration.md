# Aswath Damodaran Research Integration — stock_monitor Framework

**Purpose**: Systematically incorporate Damodaran's valuation methodologies, ERP/CRP data, corporate life cycle framework, narrative-to-numbers process, and cost-of-capital principles into our preferred metrics, implied return calculations, and quality screens.

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
| `implied_r.py` | Static/historical ERP | **Use implied ERP (forward-looking)** from Damodaran's 2-stage augmented DDM |
| `preferred_metrics.py` | No explicit cost of capital | **Compute WACC per ticker** using Damodaran's cost-of-capital framework |
| `stress_dual_pass.py` | Historical stress | **Scenario ERP shifts** (macro volatility → ERP per Lettau/Ludvigson/Wachter) |

### 1.3 Implementation
```python
# ERP as of as_of_date (pull from Damodaran's data files or API)
# US Implied ERP ≈ 4.23% (Jan 2026)
# Add CRP for non-US tickers: ERP_total = ERP_US + CRP_country
# CRP from Damodaran: CRP = (Sovereign Default Spread) * (Equity Vol / Bond Vol)
# Typical scaling: 1.2–1.5x sovereign spread
```

---

## 2. Cost of Capital — Per-Ticker WACC (Damodaran "Swiss Army Knife")

### 2.1 Framework (from costofcapital.pdf)
```
Cost of Equity = Riskfree Rate + Beta * (ERP + CRP) + [Optional: Small Cap / Liquidity Premium]
Cost of Debt = Riskfree Rate + Default Spread * (1 - Tax Rate)
WACC = Cost of Equity * (E / (D+E)) + Cost of Debt * (D / (D+E))
```

### 2.2 Key Principles for Our Use
| Principle | Application |
|-----------|-------------|
| **Dynamic, not static** | Recompute WACC quarterly as fundamentals change |
| **Business-specific, not company-wide** | For conglomerates, weight by segment (we don't have segment data — use company-level) |
| **Cost of capital �� receptacle for fears** | Don't add company-specific risk premiums to WACC; handle via probabilistic cash flows (decision trees) |
| **Small cap / liquidity premiums** | Damodaran argues these are **not justified** in modern data — exclude |
| **Marginal vs effective tax rate** | Use **marginal** for forward-looking WACC |

### 2.3 Required Inputs (Map to Our Data)
| Input | Our Source | Notes |
|-------|------------|-------|
| Risk-free rate | 10Y Treasury (FRED) | Currency-matched |
| ERP | Damodaran implied | 4.23% US Jan 2026 |
| CRP | Damodaran country tables | 0 for US |
| Beta | Regress vs SPY (5Y monthly) | Or use Damodaran's bottom-up beta by sector |
| Cost of Debt | `interest_coverage` + `debt_to_equity` | Infer from interest coverage → synthetic rating → default spread |
| Debt/Equity weights | `debt_to_equity` from fundamentals | Market value of equity preferred; book as fallback |
| Tax rate | Marginal corporate (21% US) | Adjust for NOLs if known |

### 2.4 Synthetic Rating from Interest Coverage (Damodaran Method)
| Interest Coverage | Rating | Typical Default Spread |
|-------------------|--------|----------------------|
| > 8.5x | Aaa/AAA | 0.40% |
| 6.5–8.5x | Aa/AA | 0.70% |
| 5.5–6.5x | A | 0.90% |
| 4.25–5.5x | Baa/BBB | 1.50% |
| 3–4.25x | Ba/BB | 2.50% |
| 2–3x | B | 4.00% |
| 1.5–2x | Caa/CCC | 6.00% |
| < 1.5x | Ca/CC | 10.00% |

---

## 3. Corporate Life Cycle — Stage-Aware Valuation & Quality Assessment

### 3.1 Six Stages (CorpLifeCycleLong2023.pdf)
| Stage | Characteristics | Valuation Approach | Pricing Multiples |
|-------|-----------------|-------------------|-------------------|
| **Start-up** | Negative CF, high failure risk, no history | **Option pricing / VC method** | Users, revenue (pre-revenue) |
| **Young Growth** | Revenue growth > 30%, negative/low FCF, high reinvestment | **Narrative-driven DCF** (story → numbers) | Revenue, EV/Sales |
| **High Growth** | Growth 15–30%, FCF turning positive, reinvestment peaking | **DCF with fading growth** | EV/EBITDA, P/E |
| **Mature Growth** | Growth 5–15%, stable + FCF, reinvestment moderating | **Standard DCF / Multiples** | P/E, EV/EBITDA |
| **Mature Stable** | Growth ≈ GDP, high FCF, low reinvestment | **Stable growth DCF / Dividend discount** | P/E, P/B, Div yield |
| **Decline** | Negative growth, FCF > reinvestment, asset liquidation | **Liquidation / Sum-of-parts** | P/B, EV/Assets |

### 3.2 Integration: Life Cycle Classification per Ticker
**Algorithm** (using our quarterly fundamentals):
```python
def classify_life_cycle(fundamentals_latest):
    rev_growth_3y = fundamentals['revenue_growth_3y']  # need to compute
    fcf_margin = fundamentals['fcf'] / fundamentals['revenue']
    reinvestment_rate = 1 - (fcf / ebit)  # approximate
    roic = fundamentals['roic']
    
    if rev_growth_3y > 0.30 and fcf_margin < 0:
        return "Young Growth"
    elif rev_growth_3y > 0.15 and fcf_margin < 0.05:
        return "High Growth"
    elif rev_growth_3y > 0.05 and roic > 0.15:
        return "Mature Growth"
    elif rev_growth_3y > 0.02 and fcf_margin > 0.10:
        return "Mature Stable"
    elif rev_growth_3y < 0:
        return "Decline"
    else:
        return "Unclassified"
```

### 3.3 Stage-Aware Quality/Valuation Adjustments
| Stage | Quality Threshold | Valuation Multiple | Our Score Adjustment |
|-------|-------------------|-------------------|---------------------|
| Young Growth | Narrative coherence, TAM, path to profitability | EV/Sales, EV/Users | **Reduce weight on current ROIC/earnings**; emphasize growth optionality |
| High Growth | Reinvestment efficiency (ROIIC), margin trajectory | EV/EBITDA (forward) | **ROIC less relevant**; focus on ROIIC = ΔEBIT / ΔInvested Capital |
| Mature Growth | ROIC > WACC, stable margins, FCF conversion | P/E, EV/EBITDA | **Standard quality score applies** |
| Mature Stable | ROIC > WACC, high FCF yield, capital return | P/E, P/B, Div yield | **Buffett criteria work well** |
| Decline | Asset value > going concern, liquidation proxy | P/B, EV/Assets | **Avoid unless deep value**; quality score penalized |

---

## 4. Narrative-to-Numbers — Structured DCF for Growth Names

### 4.1 Damodaran's 5-Step Process (from narrative&numbers)
1. **Develop narrative** — Simple, plausible story (e.g., "WST dominates pharma packaging, expands to biologics, margins expand")
2. **Test narrative** — History, common sense, macro consistency
3. **Convert to drivers** — Market size, market share, target margin, sales-to-capital, cost of capital
4. **Link to DCF** — Explicit forecast period, terminal value
5. **Feedback loop** — Update narrative as numbers change

### 4.2 Application: Implied Return / Reverse DCF
Our `implied_r.py` currently computes implied return from current price. **Enhance with narrative-driven scenarios:**

| Scenario | Narrative | Key Drivers | Probability |
|----------|-----------|-------------|-------------|
| **Bull** | Full TAM capture, margin expansion | Revenue CAGR +5%, margin +300bps, ROIC > WACC | 20% |
| **Base** | Steady execution, modest growth | Revenue CAGR +2%, margin flat, ROIC ≈ WACC | 60% |
| **Bear** | Competition, margin compression | Revenue CAGR -1%, margin -200bps, ROIC < WACC | 20% |

**Output**: Probability-weighted implied return, not single point estimate.

### 4.3 Young Growth Adaptation (for our universe)
- **Revenue-based DCF** for pre-profit names (use sales-to-capital, target margin)
- **Failure probability** adjustment: `Value = (1 - p_fail) * DCF_success + p_fail * Liquidation`
- **Option value** of undeveloped pipelines / TAM expansion

---

## 5. Multiples — First Principles & Fundamental Drivers

### 5.1 Damodaran's Core Insight (from multiples.pdf)
> **Every multiple is a function of fundamentals: growth, risk, cash flow characteristics.**
> 
> *PE = f(Growth, Payout, Risk)*
> *EV/EBITDA = f(Growth, ROIC, WACC, Reinvestment)*
> *EV/Sales = f(Margin, Growth, Reinvestment)*

### 5.2 Fundamental Drivers by Multiple
| Multiple | Key Drivers | Formula (Stable Growth) |
|----------|-------------|-------------------------|
| **P/E** | Growth, Payout, Risk | `P/E = (1 - g/ROE) / (r - g)` |
| **EV/EBITDA** | Growth, ROIC, WACC, Reinvestment | `EV/EBITDA = (1 - g/ROIC) * (1 - t) / (WACC - g)` |
| **EV/Sales** | Margin, Growth, Reinvestment | `EV/Sales = Margin * (1 - g/ROIC) * (1 - t) / (WACC - g)` |
| **P/B** | ROE, Growth, Risk | `P/B = (ROE - g) / (r - g)` |

### 5.3 Our Integration: Fundamental Multiple Screens
Replace static multiple thresholds with **fundamental-implied fair multiples**:

```python
def fair_pe(growth, roe, cost_of_equity, payout=None):
    """Implied P/E from fundamentals (Gordon growth)"""
    if payout is None:
        payout = 1 - growth / roe  # sustainable payout
    return payout / (cost_of_equity - growth)

def fair_ev_ebitda(growth, roic, wacc, tax_rate=0.21):
    reinvestment = growth / roic
    fcf_conversion = (1 - reinvestment) * (1 - tax_rate)
    return fcf_conversion / (wacc - growth)

def fair_ev_sales(growth, margin, roic, wacc, tax_rate=0.21):
    reinvestment = growth / roic
    fcf_conversion = margin * (1 - reinvestment) * (1 - tax_rate)
    return fcf_conversion / (wacc - growth)
```

### 5.4 Relative Valuation with Controls (Damodaran's 4 Steps)
1. **Define consistently** — Use same earnings (trailing/forward), same EBITDA definition
2. **Know distribution** — Sector + market percentiles (we have this via GICS baskets)
3. **Understand drivers** — Regress multiple vs fundamentals cross-sectionally
4. **Control for differences** — Use sector/peer regression residuals, not raw multiples

**Implementation**: Our `peer_analytics.py` should compute **fundamental-adjusted multiple z-scores**, not raw P/E ranks.

---

## 6. Margin of Safety — Post-Valuation Discipline

### 6.1 Damodaran's View (margin of safety post)
> Margin of safety is **not a substitute for risk adjustment** in valuation. It is a **discount to intrinsic value** applied *after* you have properly risk-adjusted the DCF.

### 6.2 Our Application
| Current | Damodaran-Aligned |
|---------|-------------------|
| No explicit MoS | **Require 15–25% discount to intrinsic value** before BUY |
| Implied return vs hurdle | **Implied return ≥ WACC + MoS spread** (e.g., WACC + 3–5%) |
| Quality score as proxy | **Separate quality assessment from valuation margin** |

---

## 7. Implementation Roadmap — stock_monitor Modules

### 7.1 High Priority (Immediate Value)

| Module | Change | Effort |
|--------|--------|--------|
| `implied_r.py` | Use Damodaran implied ERP (4.23%) + CRP per ticker; compute WACC per Damodaran | Medium |
| `preferred_metrics.py` | Add `wacc`, `cost_of_equity`, `cost_of_debt`, `life_cycle_stage` columns | Medium |
| `preferred_metrics.py` | Replace static multiple thresholds with **fundamental-implied fair multiples** | Medium |
| `stress_dual_pass.py` | Add ERP scenario analysis (macro vol → ERP shift) | Low |

### 7.2 Medium Priority

| Module | Change | Effort |
|--------|--------|--------|
| `peer_analytics.py` | Cross-sectional regression: multiple ~ growth + ROIC + WACC; use residuals for peer comparison | High |
| `buy_candidates.py` | Add MoS check: intrinsic value (DCF) vs price ≥ 1.15–1.25x | Medium |
| `signal_aggregator.py` | Weight signals by life cycle stage (e.g., growth signals matter more for Young Growth) | Medium |

### 7.3 Research / Long-Term

| Area | Description |
|------|-------------|
| **Narrative database** | Store structured narratives per ticker (TAM, margin path, reinvestment, risk) for audit trail |
| **Reverse DCF scenarios** | Probability-weighted implied returns (bull/base/bear) |
| **Option value for pipelines** | Real options for biotech/pharma/tech pipelines (Damodaran method) |
| **Failure probability** | Estimate from cash burn, debt maturity, altitude (for young growth) |

---

## 8. Data Sources to Ingest

| Data | Source | Frequency | Our Storage |
|------|--------|-----------|-------------|
| **US Implied ERP** | `https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html` | Monthly (Jan/Jul) | `erp_history.parquet` |
| **Country Risk Premiums** | `ctryprem*.xlsx` (Jan 2026) | Semi-annual | `crp_by_country.parquet` |
| **Sector Betas / Bottom-up Betas** | Damodaran's sector beta pages | Annual | `sector_betas.parquet` |
| **Equity Risk Premium Paper (2026)** | SSRN abstract_id=6361419 | Annual | Reference PDF |

---

## 9. Key Formulas Quick Reference (GitHub-Safe)

### 9.1 Cost of Equity
```
cost_of_equity = rf + beta * (erp_us + crp_country)
```

### 9.2 Cost of Debt (from interest coverage)
```
if interest_coverage > 8.5:    spread = 0.004
elif interest_coverage > 6.5:  spread = 0.007
elif interest_coverage > 5.5:  spread = 0.009
elif interest_coverage > 4.25: spread = 0.015
elif interest_coverage > 3:    spread = 0.025
elif interest_coverage > 2:    spread = 0.040
elif interest_coverage > 1.5:  spread = 0.060
else:                          spread = 0.100

cost_of_debt = rf + spread
after_tax_cost_of_debt = cost_of_debt * (1 - marginal_tax_rate)
```

### 9.3 WACC
```
wacc = cost_of_equity * (E / (D+E)) + after_tax_cost_of_debt * (D / (D+E))
```

### 9.4 Fair Multiples (Stable Growth)
```
fair_pe     = (1 - g/roe) / (cost_of_equity - g)
fair_ev_ebitda = (1 - g/roic) * (1 - t) / (wacc - g)
fair_ev_sales  = margin * (1 - g/roic) * (1 - t) / (wacc - g)
fair_pb     = (roe - g) / (cost_of_equity - g)
```

### 9.5 Life Cycle Classification
```
if rev_growth > 30% and fcf_margin < 0:          "Young Growth"
elif rev_growth > 15% and fcf_margin < 5%:       "High Growth"
elif rev_growth > 5% and roic > 15%:             "Mature Growth"
elif rev_growth > 2% and fcf_margin > 10%:       "Mature Stable"
elif rev_growth < 0:                             "Decline"
```

---

## 10. Validation Checklist (Before Deployment)

- [ ] ERP/CRP data pipeline automated (monthly pull from Damodaran site)
- [ ] Per-ticker WACC computed and stored in `preferred_metrics.parquet`
- [ ] Life cycle stage assigned to all 585 tickers
- [ ] Fair multiples replace static thresholds in quality screen
- [ ] Margin of safety (15–25%) enforced in `buy_candidates.py`
- [ ] Peer comparison uses fundamental-adjusted residuals
- [ ] Implied return uses probability-weighted scenarios
- [ ] Backtest: Damodaran-enhanced signals vs current signals (OOS)

---

## 11. References (Primary Sources)

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

*Generated from Damodaran primary sources (papers, data files, blog) — integrated into stock_monitor framework for implied return, quality assessment, and valuation discipline.*