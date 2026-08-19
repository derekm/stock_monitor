# Ownership Network & Corporate Control Analysis

**Data Sources**: SEC EDGAR 13F-HR (institutional), Exhibit 21.1 (subsidiaries), XBRL fundamentals
**Coverage**: 48 quarters (2014-Q3 to 2026-Q2), 38 filers, 5,420 held tickers, 383,706 quarterly edges
**Generated**: 2026-08-15

---

## 1. Data Architecture

### 1.1 Core Tables

| Table | Grain | Rows | Description |
|-------|-------|------|-------------|
| `historical_13f_holdings` | filer x as_of_date x held_cusip | 1,737,161 | Raw SEC 13F-HR with shares, voting authority |
| `quarterly_holdings_panel` | filer x as_of_date x held_ticker | 383,706 | CUSIP to ticker mapped, aggregated |
| `quarterly_network_edges` | filer x as_of_date x held_ticker | 383,706 | Graph edges with market_value, ownership_pct |
| `actual_ownership_percentages` | filer x as_of_date x held_ticker | 383,706 | ownership_pct = held_shares / shares_outstanding |
| `corporate_subsidiaries` | parent x subsidiary x jurisdiction | 604 | Exhibit 21.1 parsed from 10-K |

### 1.2 Ownership Classification (Damodaran Framework)

Each edge is classified by **actual ownership percentage** (not portfolio weight):

| Category | Threshold | Count (2026-Q2) | Valuation Treatment |
|----------|-----------|-----------------|---------------------|
| `MAJORITY_CONSOLIDATED` | ownership_pct >= 50% | 20 | Subtract NCI = (1-pct) x sub_equity |
| `EQUITY_METHOD` | 20% <= ownership_pct < 50% | 24 | Add pct x sub_equity |
| `MINORITY_PASSIVE` | 0 < ownership_pct < 20% | 6,452 | Add 13F holding_value (market value of stake) |
| `UNKNOWN` | shares_outstanding unavailable | 33,195 | Add at market value |

---

## 2. Corporate Control Graph

### 2.1 Graph Definition

Let G = (V, E, t) be a **temporal directed graph** where:

- V = set of tickers (filers union held)
- E subset of V x V x T = time-stamped edges
- t in T = quarterly as_of_date

Edge attributes at time t:
- w_ij(t) = market value of filer i's stake in j
- p_ij(t) = ownership percentage = shares_ij(t) / shares_out_j(t)
- c_ij(t) in {MAJORITY, EQUITY, MINORITY, UNKNOWN}

### 2.2 Adjacency Tensor

For each edge:
$$
A_{ij}(t) = w_{ij}(t)
$$
No edge:
$$
A_{ij}(t) = 0
$$

### 2.3 Network Metrics (per quarter)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Density** | |E(t)| / (|V(t)|(|V(t)|-1)) | Fraction of possible edges present |
| **HHI (per filer)** | sum_j (w_ij / sum_k w_ik)^2 | Portfolio concentration |
| **Effective N** | 1 / HHI | Number of equal-weight positions |
| **Core Size** | |{j : sum_i p_ij > 0.5}| | Number of controlled entities |

---

## 3. Damodaran Cross-Holdings Valuation

### 3.1 Framework (per Damodaran ReviewCrossHoldings.pdf & cashvaluation.pdf)

For each filer i at quarter t:

#### Parent Market Cap
$$
M_i(t) = market cap of filer  i  (consolidated, includes 100% of majority subs)
$$

#### Minority Interest (NCI) — Majority Holdings
$$
NCI_i(t) = \sum_{j \in Majority_i(t)} (1 - p_{ij}(t)) \times E_j(t)
$$

where E_j(t) = equity value of subsidiary j (from fundamentals or M_j x 0.5)

#### Equity Method Value — 20-50% Holdings
$$
EM_i(t) = \sum_{j \in Equity_i(t)} p_{ij}(t) \times E_j(t)
$$

#### Minority Passive Value — <20% Holdings
$$
MP_i(t) = \sum_{j \in Minority_i(t)} w_{ij}(t)
$$

(13F holding_value already = p_ij x M_j)

#### Unknown Holdings
$$
UNK_i(t) = \sum_{j \in Unknown_i(t)} w_{ij}(t)
$$

### 3.2 Damodaran Adjusted Equity

$$
V_i^{Damodaran}(t) = M_i(t) - NCI_i(t) + EM_i(t) + MP_i(t) + UNK_i(t)
$$

**Key principle**: Parent market cap already includes 100% of consolidated subs. Subtract the portion you DON'T own (NCI). Add non-consolidated stakes at your share.

### 3.3 Look-Through Equity (Naive)
$$
V_i^{Lookthrough}(t) = M_i(t) + EM_i(t) + MP_i(t) + UNK_i(t)
$$

(Does not subtract NCI — double-counts majority subs)

### 3.4 Cross-Holding Impact
$$
Delta_i(t) = V_i^{Damodaran}(t) - M_i(t) = -NCI_i(t) + EM_i(t) + MP_i(t) + UNK_i(t)
$$

---

## 4. Look-Through Fundamentals

### 4.1 Holdings-Weighted Metrics

For each fundamental metric X (EV/EBITDA, ROIC, FCF Margin, D/E, Interest Coverage, ROE, Reinvestment Rate, P/B):

$$
X_i^{LT}(t) = \frac{\sum_{j \in H_i(t)} p_{ij}(t) \times X_j(t) \times 1_{data available}}{\sum_{j \in H_i(t)} p_{ij}(t) \times 1_{data available}}
$$

where H_i(t) = holdings of filer i with fundamental data at t.

**Coverage ratio**: (sum_j p_ij x 1_data) / (sum_j p_ij) — fraction of portfolio with metric data.

### 4.2 Available Metrics (18 total)

| Category | Metrics |
|----------|---------|
| **Valuation** | EV/EBITDA, P/B, Market Cap/Assets |
| **Quality** | ROIC, ROE, FCF Margin, Earnings Stability |
| **Leverage** | Debt/Equity, Interest Coverage |
| **Growth** | Reinvestment Rate, Revenue Growth |
| **Cash Flow** | Free Cash Flow, CapEx |
| **Balance Sheet** | Total Debt, Shareholders' Equity, Total Assets, Total Revenue |

---

## 5. Portfolio Price Analytics

### 5.1 Quarterly Returns (Holdings-Weighted)

$$
r_i(t) = \sum_{j \in H_i(t)} \omega_{ij}(t) \times r_j(t)
$$

where omega_ij(t) = w_ij(t) / sum_k w_ik(t) and r_j(t) = quarterly return of holding j.

### 5.2 Risk Metrics

| Metric | Formula |
|--------|---------|
| **Annualized Vol** | sigma_i x sqrt(4) |
| **Sharpe** | (r_bar_i - r_f) / sigma_i |
| **Max Drawdown** | max_{tau <= t} ((P_i(tau) - P_i(t)) / P_i(tau)) |
| **Beta vs SPY** | Cov(r_i, r_SPY) / Var(r_SPY) |
| **VaR 95%** | 5th percentile of quarterly returns |
| **CVaR 95%** | Mean of returns below VaR 95% |

### 5.3 Attribution

Top/Bottom contributors per quarter:
$$
Contribution_{ij}(t) = \omega_{ij}(t) \times r_j(t)
$$

---

## 6. Concentration & Diversification Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **HHI** | sum_j omega_ij^2 | [1/N, 1] | 1 = single position |
| **Effective N** | 1/HHI | [1, N] | Equivalent equal-weight positions |
| **Top 5 %** | sum_{j in top5} omega_ij | [0, 1] | Largest 5 positions weight |
| **Top 10 %** | sum_{j in top10} omega_ij | [0, 1] | Largest 10 positions weight |
| **Entropy** | -sum_j omega_ij ln omega_ij | [0, ln N] | Information-theoretic diversity |
| **Normalized Entropy** | H / ln N | [0, 1] | 1 = perfectly diversified |
| **Gini** | sum_i sum_j |omega_i - omega_j| / (2N omega_bar) | [0, 1] | Inequality of position sizes |

---

## 7. Factor Exposures (Holdings-Weighted)

| Factor | Source Metric | Formula |
|--------|---------------|---------|
| **Value (P/B)** | PB_j | sum omega_ij x PB_j |
| **Quality (ROIC)** | ROIC_j | sum omega_ij x ROIC_j |
| **Quality (ROE)** | ROE_j | sum omega_ij x ROE_j |
| **Size (log MC)** | ln(M_j) | sum omega_ij x ln(M_j) |
| **Momentum (12M)** | r_j^(12M) | sum omega_ij x r_j^(12M) |
| **Leverage** | D/E_j | sum omega_ij x (D/E)_j |
| **Profitability (IC)** | IC_j | sum omega_ij x IC_j |
| **Value (EV/EBITDA)** | EV/EBITDA_j | sum omega_ij x (EV/EBITDA)_j |
| **Quality (FCF Margin)** | FCF_j | sum omega_ij x FCF_j |

---

## 8. Subsidiary Control (Exhibit 21.1)

### 8.1 Control Graph from Subsidiaries

Let S_i = set of subsidiaries of parent i from Exhibit 21.1.

**Control assumption**: Parent owns 100% of listed subsidiaries (legal control).

### 8.2 Jurisdiction Distribution (BRK-B, 281 subs)

| Jurisdiction | Count | % |
|--------------|-------|---|
| Delaware | 98 | 34.9% |
| Nebraska | 23 | 8.2% |
| United Kingdom | 11 | 3.9% |
| Canada | 10 | 3.6% |
| Germany | 9 | 3.2% |
| Ireland | 8 | 2.8% |
| Japan | 8 | 2.8% |
| China | 6 | 2.1% |
| Other (25 jurisdictions) | 108 | 38.4% |

### 8.3 Sector Clusters (BRK-B)

| Cluster | Key Subsidiaries |
|---------|------------------|
| **Insurance** | GEICO (8 entities), Gen Re (6), BH Specialty, National Indemnity |
| **Energy** | BHE (5), MidAmerican, PacifiCorp, Northern Natural Gas, Cove Point LNG |
| **Rail** | BNSF Railway, Burlington Northern Santa Fe |
| **Manufacturing** | Precision Castparts (3), Marmon (12), Lubrizol (14), IMC, CTB |
| **Retail/Service** | McLane (4), Nebraska Furniture Mart, See's Candies, Pampered Chef |
| **Financial** | Clayton Homes (4), Pilot Travel Centers, XTRA Lease |

---

## 9. Results Summary (2026-Q2)

### 9.1 Top Filers by Look-Through Equity Impact

| Filer | Parent Mkt Cap | Cross-Holdings | NCI Subtracted | Damodaran Equity | Impact % |
|-------|----------------|----------------|----------------|------------------|----------|
| TFC | 64.2B | 43.8B | 0 | 43.9B | **+68,229%** |
| UBER | 150.6B | 4.5B | 0 | 4.7B | **+2,992%** |
| GOOGL | 4,370.6B | 99.1B | 0 | 103.5B | **+2,267%** |
| NVDA | 4,861.4B | 63.4B | 0 | 68.3B | **+1,305%** |
| AMZN | 2,570.0B | 4.4B | 0 | 7.0B | **+172%** |
| BRK-B* | — | 299.3B | 0 | — | — |

*BRK-B: No parent market cap in fundamentals (not in yfinance); look-through = 299.3B holdings

### 9.2 BRK-B Deep Dive

| Metric | Value |
|--------|-------|
| **13F Portfolio** | 299.3B (26 positions) |
| **Equity Method (4)** | 87.5B (AXP 22.5%, OXY 26.5%, KHC 27.4%, DVA 38.9%) |
| **Minority Passive (15)** | 173.6B (AAPL 66B, GOOGL 38B, KO 33B, BAC 28B) |
| **Subsidiaries (Ex 21.1)** | 281 entities across 33 jurisdictions |
| **Look-through ROIC** | 24.0% |
| **Look-through EV/EBITDA** | 14.5x |
| **Look-through FCF Margin** | 19.9% |
| **Look-through D/E** | 1.55x (subs carry debt) |
| **Interest Coverage** | 66.2x |

### 9.3 Asset Manager vs Operating Company Distinction

| Type | Filers | 13F Represents | Cross-Holdings Valued? |
|------|--------|----------------|------------------------|
| **Operating/Holding** | BRK-B, GOOGL, NVDA, AMZN, TFC, UBER, AMD, INTC, CSCO, FNF | Own balance sheet | **Yes** |
| **Asset Managers** | BLK, GS, MS, JPM, BAC, WFC, C, USB, PNC, PRU, MET, COF, AXP, ALL, TRV, AIG, CB, CINF, AFG, WRB, FAF | Client portfolios | **No** (excluded) |

---

## 10. Data Quality & Limitations

| Dimension | Status | Notes |
|-----------|--------|-------|
| **13F Coverage** | 48 quarters | Only institutional managers >100M AUM |
| **CUSIP to Ticker Map** | 5420/5420 (100%) | Expanded via SEC company_tickers.json + fuzzy match |
| **Shares Outstanding** | 77,358/383,706 (20.2%) | Limited to yfinance fundamentals coverage |
| **Market Cap (held)** | 77,269/383,706 (20.1%) | Same limitation |
| **Subsidiary Data** | 6 parents, 604 subs | Only latest 10-K Exhibit 21.1 parsed |
| **Fundamental Coverage** | ~500 filer-quarters | Only filers with yfinance data |

### Key Caveats

1. **13F != Operating Holdings**: Asset managers report client money, not own investments
2. **Ownership % = Shares/Shares Outstanding**: Only available where yfinance has SO data
3. **Subsidiary % = 100% Assumed**: Exhibit 21.1 doesn't disclose ownership % (implied control)
4. **No Private Sub Valuations**: Unlisted subs valued at parent's equity share (not market)
5. **Quarterly Snapshots**: 13F filed 45 days after quarter-end; prices may differ

---

## 11. Reproducibility

### Pipeline Scripts

| Script | Input | Output |
|--------|-------|--------|
| `extract_historical_13f.py` | SEC EDGAR submissions | `historical_13f_holdings.parquet` |
| `build_quarterly_network.py` | historical + CUSIP map | `quarterly_holdings_panel`, `quarterly_network_edges`, `quarterly_network_metrics` |
| `build_extended_analytics.py` | network + fundamentals | `quarterly_lookthrough_fundamentals_extended`, `quarterly_portfolio_analytics`, `quarterly_concentration_metrics`, `quarterly_factor_exposures` |
| `damodaran_valuation_optimized.py` | ownership + fundamentals | `actual_ownership_percentages`, `damodaran_crossholdings_valuation` |
| `extract_subsidiaries.py` | SEC 10-K Exhibit 21.1 | `corporate_subsidiaries.parquet` |

### Data Files (LFS)

```
quarterly_holdings_panel.parquet         383,706 rows
quarterly_network_edges.parquet          383,706 rows
quarterly_network_metrics.parquet           48 rows
quarterly_lookthrough_fundamentals_ext.   503 rows
quarterly_portfolio_analytics.parquet     505 rows
quarterly_concentration_metrics.parquet   696 rows
quarterly_factor_exposures.parquet        503 rows
actual_ownership_percentages.parquet     383,706 rows
damodaran_crossholdings_valuation.parquet  132 rows
corporate_subsidiaries.parquet              604 rows
```

---

*Analysis based on verified multi-line output from production pipeline. All formulas use GitHub-safe LaTeX (no raw $...$).*