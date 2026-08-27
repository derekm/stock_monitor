# add_ticker.py

One-command onboarding of new tickers into the universe with full backfill.

## Why it exists (rationale)

Previously, adding a ticker required 5+ manual steps: universe add (with
hand-typed name/sector), price backfill with explicit history length,
fundamentals via EDGAR *or* yfinance fallback, market-cap recompute, then the
entire analytics pipeline. `add_ticker.py` chains those steps so a new name is
fully onboarded with max history in one command, with sensible auto-lookup of
name/sector/industry from yfinance.

## Usage

```
python add_ticker.py QSR
python add_ticker.py CAG PFE            # multiple tickers
python add_ticker.py QSR --name "Restaurant Brands" --sector "Consumer Discretionary"
python add_ticker.py QSR --no-fundamentals   # prices only
python add_ticker.py QSR --no-analytics      # skip analytics + dashboard export
```

## What it does (in order)

1. **Universe** — looks up name/sector/industry via yfinance (overridable with
   `--name/--sector/--industry`) and adds the ticker to
   `monitored_stocks.parquet` (idempotent: skips if already present).
2. **Prices** — `backfill_historical.py --period max` (full available history;
   fetches 4,000+ rows per liquid ticker back to IPO/listing).
3. **Fundamentals** — `backfill_edgar.py` for EDGAR XBRL history (deep,
   quarterly), then `update_fundamentals.py fetch-history` (yfinance quarterly
   statements) as fallback for ADRs / tickers without a CIK.
4. **Market cap** — `add_daily_marketcap.py` (close × shares outstanding).
5. **Analytics** — `run_daily_automation.py` (all ~40 jobs), then
   `export_dashboard_data.py` to refresh the dashboard.

## Outputs

- Adds rows to `monitored_stocks.parquet`, `daily_prices/`,
  `fundamentals.parquet`; then all analytics outputs (see SCHEMAS families:
  prices, fundamentals, signals, fragility, etc.).

## Related

- `manage_stocks.py` — universe CRUD if you only need the membership change.
- `backfill_historical.py` / `backfill_edgar.py` / `update_fundamentals.py` —
  the individual steps this chains.
- `run_daily_automation.py` — the analytics pipeline it triggers.
