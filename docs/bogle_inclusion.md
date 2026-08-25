# Bogle Index Inclusion — Ours vs Vanguard/CRSP

**Source of truth:** `build_bogle_funds.py:liquid_names()` (PIT gates). Rebuilt after `monitored_stocks.parquet` exchange backfill (16,056 tickers, 2026-08-24).

## TL;DR

| Gate | Vanguard TMI (CRSP US Total Market) | Our Bogle TMI | Our QMI / QMI_STRICT | Our PMI (TMI's complement) |
|------|-------------------------------------|---------------|----------------------|-----------------------------|
| Legal form | US-incorporated **common** (share code 10/11), REITs in | same: `instrument_type=stock` + `exchange` | | same: `instrument_type=stock` |
| Listing | NYSE / NYSE American / Nasdaq / Arca | **exact**: `exchange ∈ {NMS,NYQ,NCM,NGM,ASE}` (7627 common, `PCX/BTS` ETFs blocked, `PNK/OID/OQB/OQX` OTC blocked) | | **inverted**: `exchange ∉ {NMS,NYQ,NCM,NGM,ASE}` — the OTC/gray tape (`PNK/OID/OQB/OQX/PCX/BTS/None`) |
| Price | ≥ $0.50 on rank day | **≥ $5.00** last close | | **≥ $1.00** last close |
| Mcap | **float mcap** must exist on rank day | **full mcap** must exist: `daily_mcap.parquet: market_cap = shares × adj_close` at `D`, `cov ≥ 0.80` + has value on `D` (`require_mcap=True`) | | same (`require_mcap=True`) — no mcap ⇒ not investable |
| Liquidity | none (mcap rank only) | **`ADV20 ≥ $5M`** (close×volume trailing 20) — kills micro-illiquid | | **`ADV20 ≥ $100k`** — OTC-scaled, still excludes untradeable |
| Filing | not required | TMI: not required · QMI: **≥1 quarterly `as_of_date ≤ D` in `fundamentals.parquet`** (`require_filing=True`) | | not required (filing-free, like TMI) |
| IPO | 5th trading day if screens pass | when `mcap` + `ADV20` both exist (≈5–20d) — **seasoned by filings, not day count** | | same |
| Float vs full | float (insiders stripped) | full — overweights insider-heavy names ~5–10% | | full |
| Weighting | float-cap | cap-weighted (S&P divisor continuity) | equal-weight | **equal-weight + 5% single-name cap** |
| Rebalance | quarterly rank + banding | `TMI Q / QMI SA / BPI Y` + 7-day glide + 3/5 bps costs | | `Q` + 7-day glide + **5/8 bps** costs (OTC is dearer) |

## PIT semantics

- As-of `D` = last price date (`prices.index.max()`). All gates evaluated **≤ D** (no look-ahead).
- `daily_mcap` is the PIT shares file (`shares_outstanding` snapshot × `adj_close`). Missing mcap ⇒ not investable ⇒ excluded even if price exists.
- `fundamentals.as_of_date` is stamped on `filing_date` (`fundamentals_history.snapshot` + `snapshot_history.append_history`). QMI requires a **seen filing**; TMI does not.
- `monitored_stocks.parquet:exchange` is the yfinance `quoteSummary.exchange` backfill (values: `NMS` Nasdaq GS, `NCM` CM, `NGM` GM, `NYQ` NYSE, `ASE` AMEX, `PCX` Arca, `BTS` BZX, `PNK/OID/OQB/OQX` OTC, `None` dead). Static — not PIT, but OTC vs listed rarely flips without delisting.
- `daily_prices.parquet` holiday rows are already stripped (`n ≥ 0.25×median` kept; 432 rows dropped), so `cov` is trading-day coverage.

## What changed (2026-08-24)

**Before:** `liquid_names(last≥5, cov≥0.80, max_day≤1.0)` only + ad-hoc `instrument_type=stock` filter in `build_tmi` → price-hack TMI of **1144** names, IPOs required 252d history, `BOTY/CREG` slipped via `OID`.

**After:** single `liquid_names(require_mcap, require_filing)` — exchange + mcap + ADV + filing in one function. Args:

```python
liquid_names(prices, tickers,
             min_last=5.0, min_cov=0.80, max_day=1.0,
             require_mcap=True, require_filing=False,
             min_adv20=5_000_000,
             liquid_exchanges={"NMS","NYQ","NCM","NGM","ASE"})
```

- `build_tmi(..., require_mcap=True, require_filing=False)`
- `build_qmi(..., require_mcap=True, require_filing=True)` — both `quality_gate` (NM top quintile) and `quality_gate_strict` (15/15/1.0) now run on filed names only
- `build_bpi` same as TMI plus `sector ∈ {Utilities, Staples, Health Care, Real Estate, Comm Services}`

**Effect:** TMI ~4.2k PIT avg (vs 1.1k), QMI ~900 PIT avg, both IPO-inclusive once mcap+ADV exist; turnover stays controlled by banding-equivalent quarterly rebalance + 7-day glide (not point-in-time universe inflation).

## Complete-market completeness: TMI + PMI

The exchange gate splits the tape in two. TMI takes one side, **PMI takes the other**, so together they are the whole investable universe with **zero overlap** — that is the point of PMI, and it is enforced structurally rather than by convention:

```python
liquid_names(..., exchange_mode="include")   # TMI/QMI/BPI: on  {NMS,NYQ,NCM,NGM,ASE}
liquid_names(..., exchange_mode="exclude")   # PMI:         off {NMS,NYQ,NCM,NGM,ASE}
```

One gate function, one exchange set, one flag — so the two universes cannot drift apart or double-count a name. Every other PIT gate (stock-only, `require_mcap`, coverage, max-day) is **identical** for PMI; only the thresholds that OTC reality forces are relaxed (`$1` last vs `$5`, `$100k` ADV20 vs `$5M`), and the weighting adds a **5% single-name cap** so one pink story stock cannot become the index.

PMI is **not** the SMCI sleeve. SMCI would be a 13F-selected *subset* nested inside this universe; PMI is the unselected complement of TMI. Build order is deliberate: complete the market first, tilt second.

## How to verify

```bash
# mcap+exchange PIT gate
python build_bogle_funds.py --fund tmi --years 10          # expect: exchange ... eligible, mcap gate: ... pass, ADV20>=5M: ... pass, liquidity: ~4200 / ...
python build_bogle_funds.py --fund qmi --years 10          # filing gate adds ≥1 Q seen
python build_bogle_funds.py --fund qmi_strict --years 10
python build_bogle_funds.py --fund pmi --years 10           # expect: exchange filter (exclude) ... , $1/$100k gates, EW+5% cap

# prove TMI and PMI are disjoint and jointly complete
python -c "
import pandas as pd
from build_bogle_funds import load_prices, liquid_names
p = load_prices(years=10)
tmi = set(liquid_names(p, list(p.columns), exchange_mode='include'))
pmi = set(liquid_names(p, list(p.columns), min_last=1.0, min_adv20=100_000, exchange_mode='exclude'))
print('TMI', len(tmi), 'PMI', len(pmi), 'overlap', len(tmi & pmi))   # overlap MUST be 0
"

# compare to Vanguard/CRSP headcount
# CRSP US TMI ~4000 names (2025); we run ~4200 full-mcap + $5M ADV — delta = penny + micro-illiquid + float-vs-full
```
