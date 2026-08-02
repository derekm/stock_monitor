# Stock Monitor — System Overview & Investment Thesis

*Living document for the portfolio intelligence stack built across this project.*

---

## 1. What we built

A **modular, offline-capable investment operating system** around a real Robinhood book (staples, healthcare, materials/fertilizer, telecom, and a high-vol growth name), expanded into:

| Layer | Role |
|-------|------|
| **Data** | `trades`, `daily_prices`, `fundamentals` (dated), `monitored_stocks`, index levels |
| **Screens** | Value trifecta, Buffett quality, dual-pass INCLUDE_CORE, near-dual, inclusion/exclusion |
| **Risk** | Vol targeting (SMCI ≤5%), ERC / inverse-vol / GMV, robust covariance, Kelly helpers |
| **Regimes** | Rolling corr, ALLPAIRS, crisis corr breakdown, HMM/Kalman/VAR (earlier threads) |
| **Forecast** | Granite TTM-style path, exogenous features, anomaly scans, microservice |
| **Indexes** | Fertilizer, defensive value, personal, growth/tech (+ Starlink/launch/aerospace) |
| **Allocation** | Factor rotation (quality/value/dual/low-vol/dividend), tail hedges, Black–Litterman |
| **Ops** | Alerts (price + fundamentals), daily automation, DuckDB-Wasm dashboard, analytics API |

It is designed so **screens, risk budgets, and regime signals** can change weights over time—not a static “buy the trifecta and forget.”

---

## 2. What we are analyzing (and why)

### Valuation (trifecta)
- **EV/EBITDA ≤ 9** — operating value not overly rich  
- **P/B ≤ 1.5** — modest premium to book  
- **MktCap/Assets ≤ 0.5** — assets not fully priced away  

*Basis:* classic value; cheapness is necessary but not sufficient.

### Quality (Buffett-style)
- **ROE ≥ 15%, ROIC ≥ 15%**, **D/E ≤ 1**  
- DuPont: prefer ROE from **operations**, not leverage  
- Earnings stability as a predictability proxy  

*Basis:* durable economic returns compound; leverage-inflated ROE fails in stress.

### Dual-pass (INCLUDE_CORE)
All six legs. Empirically rare: quality is expensive; cheap names often have weak ROIC. Stress tests show the gate is stable (tight → 0 names; base → ~5 regional bank/AM names; relaxed → teens).

### Risk & sizing
- **Volatility targeting** and hard **SMCI ≤ 5%** (gap/AI risk ≠ short-window σ)  
- **ERC** equalizes risk contribution; **GMV** minimizes vol; **inv-vol** is a simple diagonal ERC  
- Robust covariance (Ledoit–Wolf) stabilizes optimizers  
- Rolling vol, beta, max DD, CVaR on screens and stress tables  

### Correlation & regimes
- Rolling windows (21/63/126), ALLPAIRS history, crisis vs calm  
- In factor-structured data: calm pairwise ~0.15, crisis ~0.45+, sector crisis higher still  
- *Implication:* diversification **fails when you need it**; hedges and cash buffers matter  

### Factor rotation (defense)
Sleeves: quality, value, dual, low-vol, dividend ETFs, full defensive index.  
Risk-off → low-vol + dividend ETFs; risk-on → quality/dual; value tilt when value lags.

### Tail hedges
Cash 10–20%, low-vol tilt, defensive ETF blend, put-proxy, **tail_combo**.  
Cash and combo cut vol and max DD most cleanly in sample.

### Growth satellite
Growth/tech + Starlink supply + launch (RKLB, SPCX, …) is a **capped satellite**, not the core. ERC/GMV inside the sleeve; portfolio-level SMCI/vol caps still bind.

---

## 3. What these goals/thresholds can achieve

| Objective | Mechanism |
|-----------|-----------|
| Avoid permanent capital loss | Quality + leverage limit + tail hedges |
| Buy under-earning assets | Trifecta + near-dual watchlist |
| Don’t overpay for quality | Dual-pass rarity forces “fair price” discipline |
| Contain idiosyncratic blow-ups | Name caps, vol targeting, satellite budget |
| Adapt to regimes | Corr/vol signals → factor rotation |
| Stay implementable | Parquet + CSV + dashboard + one daily script |

**They cannot:** guarantee outperformance, replace judgment on fraud/regulation/technology shifts, or make synthetic history into live edge. Thresholds are **policy**, not truth.

---

## 4. Beating the S&P 500 over ~10 years — a realistic design

S&P 500 is a **large-cap, momentum-tolerant, quality-growth-heavy** benchmark. Beating it after costs requires a *repeatable edge* and *risk discipline*, not constant high beta.

### Proposed dynamic fund architecture

1. **Core (60–75%) — Defensive compounders & dual/near-dual value**  
   - Dual-pass and INCLUDE_QUALITY at fair-or-better prices  
   - Trifecta value with acceptable ROIC  
   - Defensive ETFs (SCHD, VIG, XLP, XLU, USMV) as ballast  
   - Rebalance on screen changes + quarterly fundamentals  

2. **Flex value (10–20%) — Cyclical QARP**  
   - Energy/refiners, steel, select financials when near-dual and mid-cycle  
   - Factor rotation overweight when value lags quality  

3. **Growth satellite (5–15%) — Hard-capped**  
   - Growth/tech + aerospace/Starlink chain  
   - **SMCI ≤ 5% of total fund**; sleeve vol budget  
   - Only add on drawdowns or improving quality metrics  

4. **Hedge / liquidity (5–15%)**  
   - Cash or short-duration ballast in high-corr/high-vol regimes  
   - Optional put-proxy or min-vol overlay when crisis corr spikes  

### Process (quarterly + monthly)
- Monthly: vol regime, rolling corr, factor weights, alerts  
- Quarterly: fundamentals history, dual/near-dual, inclusion tables, DuPont  
- Continuous: price/fundamental alerts, SMCI/vol caps  

### Why this can work vs SPX (not a promise)
- **Avoid the left tail** of single-name and high-corr crises  
- **Buy dollars for 70¢** when trifecta + decent ROIC appear  
- **Pay up only for durable ROIC**, sized by quality_score  
- **Don’t let a satellite eat the fund** (the historical failure mode of “intelligent” growth sleeves)  

Edge is **process + discipline**, not a single factor.

---

## 5. Future work

### Better quant
- Live fundamentals API (replace seeds/backfill noise)  
- Real crisis labels (2008, 2020, 2022) on long history  
- Options-based hedges (put spreads) vs put_proxy  
- Transaction-cost and tax-aware rebalancing  
- Walk-forward dual-pass / rotation (no look-ahead)  
- Shrinkage + Black–Litterman views from screens automatically  
- Partial-duration or inflation factors for true multi-asset defense  

### Deeper integration
- Single “decision engine” reading inclusion + risk + regime → target weights  
- Dashboard: one-click “proposed trades” vs holdings  
- Unify Fisher, Granite forecast, and factor rotation on shared calendar  
- Alerting when dual-pass set changes or crisis corr regime flips  
- Paper-trade ledger with slippage vs SPX/SPY  

### Data integrity
- Price pipeline health checks (the independent-synthetic corr ≈ 0 failure mode)  
- Factor-structured or vendor data as default for regime research  

---

## 6. Daily command

```bash
python run_daily_automation.py
# or selective:
python run_daily_automation.py --only inclusion,stress,rolling_corr,tail_hedge,export
```

Dashboard: `index.html` + `dashboard_data/data.json`  
Services: `analytics_service.py`, `granite_service.py`
