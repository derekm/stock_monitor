# Ownership Network Implementation Plan — Section 4 A-F Steps

## Overview
This document outlines the A-F steps for implementing a vectorized ownership network that tracks what stocks and securities public companies own, enabling calculation of daily EV/EBITDA from owned securities and analysis of global corporate control.

## A. Foundation: Detailed Holdings Extraction Infrastructure
**Goal**: Build robust infrastructure to extract issuer-specific holdings from SEC XBRL filings.

**Completed**:
- ✅ Created `extract_detailed_holdings.py` - Parses inline XBRL from 10-K/10-Q filings
- ✅ Created `debug_xbrl.py`, `debug_xbrl2.py`, `debug_facts.py` - Tools for XBRL analysis
- ✅ Created `scan_holdings_tags.py` - Scans for relevant XBRL tags
- ✅ Created `examine_xbrl_contexts.py` - Examines XBRL contexts for issuer information
- ✅ Built CIK ticker mapping system using SEC's company_tickers.json
- ✅ Established SEC data fetching patterns with proper User-Agent and rate limiting

**Next**: Refine XBRL parser to correctly associate dimensions with investment concepts and extract issuer identifiers.

## B. Extract Issuer-Specific Holdings from Filings
**Goal**: Extract detailed holdings data showing which specific securities each company owns.

**Completed**:
- ✅ Parser extracts contexts, units, and facts from XBRL filings
- ✅ Filer identification via ticker → CIK mapping
- ✅ Filing retrieval: 10-K/10-Q submission filtering and download
- ✅ Context parsing: periods, entities, and segments
- ✅ Fact extraction: numeric and non-numeric XBRL facts
- ✅ Holding identification: filters for investment-related concepts
- ✅ Issuer identification: attempts to extract issuer CIK from segment dimensions

**Next**: 
- Improve dimension-to-issuer mapping (many dimensions don't use standard patterns)
- Handle cases where issuer is implied rather than explicit in dimensions
- Extract actual share counts and market values from XBRL facts
- Validate extraction against known holdings (e.g., Berkshire's Apple stake)

## C. Convert Holdings to Market Values
**Goal**: Convert raw holding values (shares or USD) to consistent market values using daily prices.

**Completed**:
- ✅ `daily_prices/` available as price feed
- ✅ Built framework for converting shares × price = market value
- ✅ Established additive merge patterns for timeseries data

**Next**:
- Implement conversion logic in holdings extractor:
  - If unit is shares: lookup held ticker's price as of filing date/period end
  - If unit is USD: use raw value directly
  - If unit is pure/ratio: determine appropriate conversion or skip
- Aggregate holdings by (filer_ticker, as_of_date, held_ticker)
- Output `holdings_detailed.parquet` with shares and market_value columns

## D. Build Ownership Network Structure
**Goal**: Construct network graph representing ownership relationships between companies.

**Completed**:
- ✅ `ownership_network_edges.parquet` and `.nodes.parquet` from aggregated holdings data
- ✅ `build_ownership_network.py` - Creates edges from filer to holding categories
- ✅ Demonstrated network structure concept using investment holdings panel

**Next**:
- Replace simplified category-based network with true security-level network
- Edges: `filer_ticker → held_ticker` weighted by market value of holding
- Nodes: all companies with attributes (sector, market cap, etc.)
- Implement using pandas/networkx or similar for efficient computation
- Output: `ownership_network_edges.parquet` (filer, held, market_value, date) and `ownership_network_nodes.parquet` (ticker, sector, market_cap, etc.)

## E. Calculate Network Metrics and Look-Through EV/EBITDA
**Goal**: Derive actionable insights from the ownership network.

**Completed**:
- ✅ Framework for calculating holdings-to-market-cap ratios
- ✅ Sample joins with fundamentals data demonstrated
- ✅ Understanding of look-through EV/EBITDA concept (weighted average of holdings' EV/EBITDA)

**Next**:
- Calculate network metrics:
  - Out-degree: number of securities each company holds
  - In-degree: number of companies holding each security  
  - Market value concentration: % of portfolio in top 5/10 holdings
  - Sector exposure: weighted exposure to different GICS sectors
  - Eigenvector centrality: influence in ownership network
- Calculate look-through EV/EBITDA:
  - For each filer, compute weighted average EV/EBITDA of held securities
  - Weight by market value of each holding
  - Store as `look_through_ev_ebitda.parquet` additive column
- Calculate look-through revenue growth and other fundamentals similarly

## F. Integration, Validation, and Application
**Goal**: Integrate ownership network with existing framework and validate utility.

**Completed**:
- ✅ Additive data philosophy maintained - no overwriting existing parquet files
- ✅ DATE-native convention followed throughout
- ✅ Verify-by-running principle applied to all scripts
- ✅ Integration with Damodaran quality screens and fundamental analysis pipeline

**Next**:
- Join ownership network metrics with `fundamentals.parquet`:
  - Add columns: `holdings_market_value`, `holdings_count`, `holdings_sector_entropy`
  - Add look-through columns: `lt_ev_ebitda`, `lt_revenue_growth`, `lt_roic`
- Enhance quality screens with ownership network factors:
  - Consider concentration risk in quality assessment
  - Look for synergies between owned businesses
- Validate with known cases:
  - Berkshire Hathaway's Apple ownership should show in BRK-B's network
  - Microsoft's LinkedIn ownership (pre-acquisition) should be visible
  - Check if conglomerates show expected diversification patterns
- Build factors/signals from ownership network data:
  - Ownership concentration factor
  - Look-through value factor
  - Corporate control exposure metrics
- Create visualization/dashboard components for ownership network exploration

## Deliverables
By completing steps A-F, we will have:
1. `holdings_detailed.parquet` - issuer-level holdings from XBRL filings
2. `holdings_panel.parquet` - aggregated holdings per filer per date
3. `ownership_network_edges.parquet` - filer → held_security network edges
4. `ownership_network_nodes.parquet` - network node attributes
5. Enhanced `fundamentals.parquet` with ownership-derived columns
6. Validation report showing accuracy against known holdings
- Documentation: `docs/ownership_network.md` explaining the system

## Implementation Notes
- All data remains additive: new parquet files created, existing ones never overwritten
- All dates maintained as datetime.date objects (DATE-native convention)
- Scripts designed for verify-by-running: testable on small samples before scaling
- SEC rate limiting respected: 0.12s between requests, proper User-Agent header
- CIK mappings maintained and extended as needed for missing/override cases
