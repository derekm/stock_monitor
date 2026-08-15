# Ownership Network & Corporate Control Analysis

**Repository**: `stock_monitor`  
**Last Updated**: 2026-08-15  
**Data Coverage**: 38 institutional 13F-HR filers, 48 quarters (2014-Q3 → 2026-Q2), 1.74M raw holdings

---

## 1. Data Architecture

### 1.1 Data Sources

| Source | Form Type | Coverage | Granularity |
|--------|-----------|----------|-------------|
| SEC EDGAR | 13F-HR | 38 institutional managers | Issuer-level (CUSIP → ticker) |
| SEC EDGAR | 10-K/10-Q | 8 financial companies | Asset-class level (XBRL dimensions) |
| SEC EDGAR | Exhibit 21.1 | 12 test companies | Subsidiary lists (jurisdiction) |

### 1.2 Core Tables

| Table | Grain | Rows | Key Columns |
|-------|-------|------|-------------|
| `historical_13f_holdings.parquet` | filer × as_of_date × held_cusip | 1,737,161 | filer_ticker, as_of_date, held_cusip, held_shares, held_value_thousands |
| `quarterly_holdings_panel.parquet` | filer × quarter × held_ticker | 12,312 | filer_ticker, as_of_date, held_ticker, market_value, ownership_pct |
| `quarterly_network_edges.parquet` | filer → held (per quarter) | 12,312 | as_of_date, filer_ticker, held_ticker, market_value, ownership_pct |
| `quarterly_network_metrics.parquet` | quarter | 46 | as_of_date, n_filers, n_held, total_value, hhi, network_density, core_size |
| `quarterly_lookthrough_fundamentals.parquet` | filer × quarter | 419 | filer_ticker, as_of_date, lt_ev_ebitda, lt_roic, lt_fcf_margin, lt_debt_to_equity |

### 1.3 Universe Membership

| Universe | Count |
|----------|-------|
| Fundamentals tickers | 601 |
| Daily prices tickers | 602 |
| SEC CIK map | 10,387 |
| **13F-HR filers in universe** | **16** (2.7%) |
| **With any holdings data** | **24** (4.0%) |

> **Key limitation**: SEC only requires issuer-level disclosure for institutional investment managers (>$100M AUM). Operating companies (AAPL, MSFT, NVDA, etc.) do not file 13F-HR. Their equity holdings are not publicly available at issuer granularity.

---

## 2. Network Construction

### 2.1 Bipartite Holdings Graph

We construct a **bipartite directed graph** $G = (V_f \cup V_h, E)$ where:

- $V_f$: set of filers (institutional managers)
- $V_h$: set of held securities (public equities)
- $E \subseteq V_f \times V_h$: ownership edges

Each edge $(i, j) \in E$ carries weight $w_{ij}^{(t)}$ = market value of filer $i$'s position in security $j$ at quarter $t$.

**Ownership percentage** (normalized per filer per quarter):

$$p_{ij}^{(t)} = \frac{w_{ij}^{(t)}}{\sum_{k \in V_h} w_{ik}^{(t)}}$$

### 2.2 Filer Overlap Network (Projection)

Project onto filers to capture **common ownership** structure:

$$S_{ij}^{(t)} = \sum_{k \in V_h} \min(p_{ik}^{(t)}, p_{jk}^{(t)}) \quad \text{(overlap value)}$$

$$\text{cosine}_{ij}^{(t)} = \frac{\sum_k p_{ik}^{(t)} p_{jk}^{(t)}}{\sqrt{\sum_k (p_{ik}^{(t)})^2} \sqrt{\sum_k (p_{jk}^{(t)})^2}}$$

$$\text{Jaccard}_{ij}^{(t)} = \frac{|\{k: p_{ik}^{(t)} > 0 \land p_{jk}^{(t)} > 0\}|}{|\{k: p_{ik}^{(t)} > 0 \lor p_{jk}^{(t)} > 0\}|}$$

### 2.3 Network Control (Vitali et al. 2011 Adaptation)

Following *Vitali, Glattfelder & Battiston (2011)* "The network of global corporate control" (arXiv:1107.5728), we adapt their **network control** methodology for institutional holdings.

**Threshold control model** (original):
$$C_{ij} = \begin{cases} 1 & \text{if } W_{ij} > 0.5 \\ 0 & \text{otherwise} \end{cases}$$

**Proportional control model** (our adaptation for 13F-HR):
$$C_{ij}^{(t)} = p_{ij}^{(t)} \in [0, 1]$$

**Network control** (solving $C^{\text{net}} = C + C \cdot C^{\text{net}}$ with PageRank-style damping):
$$C^{\text{net}} = (I - \alpha C)^{-1} C \quad \text{where } \alpha = 0.85$$

Total network control for filer $i$:
$$c_i^{\text{net}} = \sum_j C_{ij}^{\text{net}}$$

**Control value** (economic value influenced):
$$v_i^{\text{net}} = \sum_j C_{ij}^{\text{net}} \cdot V_j$$
where $V_j$ = market capitalization of held security $j$.

---

## 3. Bow-Tie Decomposition

Following the **bow-tie topology** of Vitali et al. (2011), we decompose the filer overlap network:

```
                    IN
                     ↓
            OUT ← SCC (core) → T&T
```

- **SCC (Strongly Connected Component)**: Core where every filer reaches every other via common holdings
- **IN**: Filers that can reach core but not reachable from core
- **OUT**: Filers reachable from core but cannot reach core
- **T&T (Tubes & Tendrils)**: Filers connected to LCC but not to core

**Our findings**: The filer overlap network forms a **single giant SCC** (all 16 filers in core), indicating a tightly knit "super-entity" of institutional managers — consistent with Vitali et al.'s finding that 73% of top control-holders are financial institutions.

---

## 4. Concentration Metrics

### 4.1 Herfindahl-Hirschman Index (HHI)

$$HHI^{(t)} = \sum_{i \in V_f} \left( \frac{v_i^{(t)}}{V_{\text{total}}^{(t)}} \right)^2$$
where $v_i^{(t)} = \sum_j w_{ij}^{(t)}$ = total holdings value of filer $i$.

### 4.2 Top-K Concentration

$$C_k^{(t)} = \frac{\sum_{i \in \text{top-}k} v_i^{(t)}}{V_{\text{total}}^{(t)}}$$

### 4.3 Network Density

$$\rho^{(t)} = \frac{|\{ (i,j) : \text{cosine}_{ij}^{(t)} > \theta \}|}{|V_f| (|V_f| - 1)}$$

---

## 5. Look-Through Fundamentals

For each filer $i$ at quarter $t$, compute **holdings-weighted averages** of fundamental metrics:

$$\text{LT-Metric}_i^{(t)} = \frac{\sum_{j \in H_i^{(t)}} w_{ij}^{(t)} \cdot \text{Metric}_j^{(t)}}{\sum_{j \in H_i^{(t)}} w_{ij}^{(t)}}$$

where $H_i^{(t)}$ = set of held tickers with available fundamentals at $t$.

**Metrics computed**:
- $lt\_ev\_ebitda$: Enterprise Value / EBITDA
- $lt\_roic$: Return on Invested Capital
- $lt\_fcf\_margin$: Free Cash Flow / Revenue
- $lt\_debt\_to\_equity$: Total Debt / Shareholders' Equity
- $lt\_interest\_coverage$: EBIT / Interest Expense

---

## 6. Corporate Subsidiary Data

### 6.1 Source: Exhibit 21.1

SEC Form 10-K requires **Exhibit 21.1** — "Subsidiaries of the Registrant" — listing all significant subsidiaries with jurisdiction of incorporation.

### 6.2 Extraction Method

1. Fetch latest company-filed 10-K (accession starts with CIK prefix)
2. Parse filing directory for Exhibit 21.1 HTML file
3. Extract table rows: (Subsidiary Name, Jurisdiction)
4. Map to parent ticker/CIK

### 6.3 Example: Microsoft (MSFT)

| Subsidiary | Jurisdiction |
|------------|--------------|
| Microsoft Ireland Research Unlimited Company | Ireland |
| Microsoft Global Finance Unlimited Company | Ireland |
| Microsoft Ireland Operations Limited | Ireland |
| Microsoft Online, Inc. | United States |
| LinkedIn Corporation | United States |
| Activision Blizzard, Inc. | United States |
| ... | ... |

**Status**: Extraction framework built (`extract_subsidiaries.py`), parsing works for MSFT, AAPL formats. Need to handle format variations across companies.

---

## 7. Key Results Summary (as of 2026-Q2)

### 7.1 Top Filers by Control Value

| Filer | Holdings | Control Value | Network Control |
|-------|----------|---------------|-----------------|
| **BLK** | $2.57T | **$1.38T** | 0.539 |
| **JPM** | $560B | **$304B** | 0.542 |
| **MS** | $541B | **$289B** | 0.535 |
| **GS** | $306B | **$165B** | 0.538 |
| **BAC** | $306B | **$161B** | 0.528 |

### 7.2 Most Held Securities (by total value)

| Ticker | Total Held Value | Filers |
|--------|------------------|--------|
| **NVDA** | $655B | 11 |
| **AAPL** | $645B | 12 |
| **GOOGL** | $586B | 11 |
| **MSFT** | $400B | 13 |

### 7.3 Bridge Securities (Highest Betweenness)

| Ticker | Betweenness | Filers | Value |
|--------|-------------|--------|-------|
| **BAC** | 0.0876 | 11 | $80B |
| **KHC** | 0.0294 | 12 | $11B |
| **MSFT** | 0.0115 | 13 | $400B |

---

## 8. Temporal Evolution

| Period | Filers | Network Density | Core Size |
|--------|--------|-----------------|-----------|
| 2015-2019 | 1-3 | 0.00 | 1 |
| 2020-Q2 | ~10 | 0.05 | 2-3 |
| 2021-Q2 | ~20 | 0.15 | 5-8 |
| 2023-2026 | 24 | 0.35+ | 16 (full) |

The institutional ownership network has **densified significantly** over the last decade, with the core "super-entity" expanding from a handful of managers to the full cohort.

---

## 9. Implementation Files

| File | Purpose |
|------|---------|
| `extract_detailed_holdings.py` | Multi-form parser (13F-HR + 10-K/10-Q XBRL) |
| `parse_13f_hr.py` | Dedicated 13F-HR information table parser |
| `build_holdings_panel.py` | Convert raw holdings → quarterly panel |
| `build_ownership_network.py` | Network edges, metrics, look-through |
| `extract_historical_13f.py` | Historical 13F-HR extraction (all quarters) |
| `build_quarterly_network.py` | Vectorized quarterly panel builder |
| `implement_vitali.py` | Direct Vitali et al. techniques |
| `implement_vitali_adapted.py` | Institutional-adapted Vitali techniques |
| `extract_subsidiaries.py` | Exhibit 21.1 subsidiary extractor |

---

## 10. Limitations & Future Work

### Current Limitations

1. **CUSIP→ticker mapping**: Only 58 mappings (3% identification rate)
2. **Operating company holdings**: Not publicly available at issuer level
3. **Subsidiary extraction**: Format variations require per-company handling
4. **13F-HR amendments**: Currently skipped (could double-count if not handled)

### Planned Extensions

1. **N-PORT filings**: Mutual fund/ETF holdings (monthly, more granular)
2. **SC 13D/G**: Beneficial ownership >5% (event-driven, sparse)
3. **Cross-border ownership**: Via Orbis/company registry data
4. **Option-implied holdings**: From 13F-HR put/call disclosures
5. **Dynamic control**: Time-varying network control with quarterly rebalancing

---

## References

1. **Vitali, S., Glattfelder, J.B., & Battiston, S.** (2011). *The network of global corporate control*. PLoS ONE 6(10): e25995. arXiv:1107.5728
2. **SEC Form 13F-HR**: Quarterly institutional investment manager holdings
3. **SEC Form 10-K Exhibit 21.1**: Subsidiaries of the registrant
4. **SEC EDGAR API**: Company facts, submissions, filing documents