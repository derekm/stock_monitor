# S&P 500 Index Effect Research — "What Happened to the Index Effect?" (Preston & Soe, 2021)

**Source**: S&P Dow Jones Indices Research Paper, September 2021  
**Authors**: Hamish Preston (Director, U.S. Equity Indices) & Aye Soe (Managing Director, Core & Multi-Asset Indices)  
**PDF**: `https://www.spglobal.com/spdji/en/documents/research/research-what-happened-to-the-index-effect.pdf`  
**Article**: `https://www.spglobal.com/spdji/en/research/article/what-happened-to-the-index-effect-a-look-at-three-decades-of-sp-500-adds-and-drops/`

---

## Executive Summary

The **"index effect"** — abnormal returns around S&P 500 additions/deletions between announcement and effective dates — has undergone a **structural decline and near-disappearance** over three decades (1995–2021).

| Period | Additions Return | Deletions Return | Spread (Add–Del) |
|--------|------------------|------------------|------------------|
| **1995–1999** | **+8.32%** | **−9.58%** | **17.9 pp** |
| **2000–2010** | +2.1% | −3.2% | 5.3 pp |
| **2011–2021** | **−0.3%** | **+0.8%** | **−1.1 pp** (reversed!) |

**Key finding**: The index effect has not just weakened — it **vanished and slightly reversed** in the most recent decade.

---

## Why It Disappeared: The Liquidity Hypothesis

### Mechanism (Pre-2010)
1. S&P 500 index funds held **vastly more AUM** than mid/small-cap funds
2. When a small-cap stock graduated to S&P 500:
   - S&P 500 funds **must buy** (large forced demand)
   - Mid/small-cap funds **must sell** (small forced supply)
   - **Massive imbalance → price spike**
3. Reverse for deletions

### What Changed (2010+)
| Factor | 2007 | 2020 | Change |
|--------|------|------|--------|
| **ETF AUM (non-S&P 500)** | $807B | ~$8T | **10x** |
| **S&P MidCap 400 AUM** | — | 12x growth | |
| **S&P SmallCap 600 AUM** | — | 35x growth | |

**Result**: When a stock graduates from MidCap 400/SmallCap 600 → S&P 500:
- S&P 500 funds buy (large AUM)
- Mid/Small funds sell (now **much larger AUM**)
- **Supply ≈ Demand → no price pressure**

---

## Critical Nuance: "Outside" Additions Still Show the Effect

| Addition Type | Index Effect (2011–2021) | Example |
|---------------|--------------------------|---------|
| **Graduates** (Mid/Small → S&P 500) | **Gone / Reversed** | Typical rebalance |
| **Outsiders** (Large caps not in S&P 1500) | **Persists** | **Tesla (2020)**: +70% announcement→inclusion |

**Why outsiders differ**: No mid/small-cap funds to sell — only S&P 500 buying pressure.

---

## Implications for Our Framework

### 1. **Do NOT rely on index effect for alpha**
- Historical backtests using pre-2010 index effect will **overstate returns**
- Current期: buying announced additions = negative expectancy

### 2. **S&P 500 membership changes in our PIT data**
- Our `sp500_membership.parquet` + `sp500_changes_merged.parquet` capture **all additions/deletions 1957–present**
- Can test index effect **by era** using our PIT membership panel
- Must control for: graduate vs. outsider, era, market regime

### 3. **Liquidity matters more than passive AUM growth**
- Passive growth alone would predict **stronger** index effect
- The offset is **liquidity growth in adjacent indexes** (MidCap 400, SmallCap 600)
- Our `daily_prices.parquet` has volume data → can compute liquidity metrics

### 4. **Tesla-type events are the exception**
- Large outsider additions (BRK.B, TSLA, potentially future mega-caps)
- Still create temporary dislocations
- Our `inclusion_signals.py` / `shock_ride.py` may capture these

---

## Testable Hypotheses Using Our Data

```python
# 1. Index effect by era (using our PIT membership + daily_prices)
def index_effect_analysis():
    changes = pd.read_parquet('sp500_changes_merged.parquet')
    prices = pd.read_parquet('daily_prices.parquet')
    
    for era in [(1995,1999), (2000,2010), (2011,2021)]:
        era_changes = changes[(changes['event_date'].dt.year >= era[0]) & 
                              (changes['event_date'].dt.year <= era[1])]
        # Compute announcement→effective returns for additions vs deletions
        # Compare graduates vs outsiders (need to classify)

# 2. Liquidity proxy: avg daily volume / market cap in year before addition
#    Test if higher liquidity → smaller index effect

# 3. Cross-index AUM proxy: MidCap 400 + SmallCap 600 constituent count growth
#    Our membership panel tracks all three indexes over time

# 4. Post-inclusion reversal (the "round trip")
#    Morningstar shows additions often give back gains post-effective-date
#    Test hold periods: 1w, 1m, 3m, 6m, 12m post-effective
```

---

## Key Quotes (for reference)

> "The passive investing ecosystem is evolving, with index rebalance at the heart of it." — Preston & Soe

> "Investors can no longer take for granted what used to be a fairly reliable pattern." — Mark Hulbert, MarketWatch (citing the paper)

> "The amount of money indexed to the S&P MidCap 400 and S&P SmallCap 600 indexes increased about 12-fold and 35-fold, respectively, over the 20 years through 2021." — Morningstar summary

---

## Integration with Our SP500 History Build

Our `build_sp500_history.py` already produces:
- `sp500_changes_merged.parquet` — all add/remove events with dates, reasons
- `sp500_membership.parquet` — daily PIT membership (joinable with `daily_prices`)

**Next step**: Add a research script `index_effect_analysis.py` that:
1. Classifies each addition as **graduate** (from MidCap 400/SmallCap 600) vs **outsider**
2. Computes announcement→effective and effective→post returns
3. Outputs era-by-era statistics matching Preston & Soe
4. Tests liquidity / AUM-growth explanations

---

## ������ CRITICAL CAVEAT: Missing Original 1957 Constituents

### The Gap
| Source | Original 1957-03-04 Members | Notes |
|--------|----------------------------|-------|
| **tickerleague.com** (official S&P announcements) | **205** | Only 205 adds on 1957-03-03/04 |
| **Wikipedia** (curated changes table) | **205** | Only 205 adds on 1957-03-04 |
| **Historical fact** (Siegel 2014, NBER, S&P records) | **500** | 425 industrials + 60 utilities + 15 railroads |

**Both our primary public sources are missing 295 original constituents (59% of the launch index).**

### Why the Sources Are Incomplete
The original 500 list was published in the **Standard & Poor's Security Price Index Record for 1957** — scattered across **91 industry subgroups** (e.g., "Confectionery: American Chicle, Hershey Chocolate, Sweets, Wrigley"). This print publication:
- Is **not digitized freely** (HathiTrust/Google Books have only "search-only" 1957 edition)
- Requires **WRDS/COMPUSTAT/CRSP** institutional access for the full constituent list
- Was the source used by Siegel (2014) and NBER researchers

### Impact on Our Data
| Artifact | Status | Note |
|----------|--------|------|
| `sp500_changes_merged.parquet` | **Accurate for documented changes** | Only 205/500 launch members have public add records |
| `sp500_membership.parquet` | **Incomplete before 1964** | Only 205 tickers on 1957-03-04; first documented change after launch is 1964 |
| `sp500_constituents_validated.parquet` | **Current only** | 53 survivors with `date_added=1957-03-04` (matches Wikipedia) |

### Recommended Usage
```python
# When using sp500_membership.parquet for pre-1964 research:
# - KNOWN: 205 members on 1957-03-04 (documented in public sources)
# - ACTUAL: 500 members on 1957-03-04 (per S&P historical records)
# - GAP: 295 members with no public add/remove trail
# - FIRST DOCUMENTED CHANGE: 1964 (per tickerleague)
# - Use with caution for pre-1964 index composition studies
```

---

## References

1. **Primary**: Preston, H. & Soe, A.M. (2021). "What Happened to the Index Effect? A Look at Three Decades of S&P 500 Adds and Drops." S&P Dow Jones Indices Research.
2. **Coverage**: MarketWatch (Hulbert, 2021), Morningstar (2022), Investopedia (2022), Harvard/Greenwood & Sammon (2024 SSRN-4294297)
3. **Historical Composition**: Siegel, J. (2014). "The Long-term Return on the Original S&P 500 Firms." Wharton/Rodney White Center.
4. **Original 1957 List**: Standard & Poor's Security Price Index Record, 1957 edition (91 industry subgroups, 500 companies).
5. **Our Data**: `sp500_changes_merged.parquet` (1,534 events), `sp500_membership.parquet` (9.36M PIT rows), `daily_prices.parquet` (OHLCV)

---

*Added to stock_monitor research library: 2026-08-13*  
*Updated with missing constituents caveat: 2026-08-13*