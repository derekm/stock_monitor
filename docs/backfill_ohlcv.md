# backfill_ohlcv.py — backfill full OHLCV history for the entire universe

## Why it exists (rationale)

The stored `daily_prices.parquet` is close+volume only for almost all tickers
(OHLC coverage ~0.5%): the daily history came from a close-only source, and the
Polygon flat-files that carry true open/high/low are blocked on this plan.
Full OHLCV matters for:
- true VWAP (open/high/low + volume), which the statistical profiler can only
  approximate from close+volume today
- ATR / chandelier stops and the structural gate (turtle / hybrid modes), which
  currently degrade to close-only approximations
- Granite multivariate panels and gap analysis (which need true open/high/low)

`backfill_ohlcv.py` fills the gap with **yfinance's full OHLCV history**
(open/high/low/close/volume + Adj Close), which goes back decades per ticker.

## What it does

For each ticker:
1. fetches `period='max'` daily OHLCV via yfinance (`auto_adjust=False`, so the
   raw Open/High/Low/Close/Volume are as-traded; Adj Close is stored as
   `adj_close`)
2. merges **strictly additively** into `daily_prices.parquet` — it only **fills**
   `open/high/low` where they are currently NaN in existing rows and **adds**
   brand-new (date, ticker) rows the table lacks. It **never overwrites**
   existing `close`, `volume`, `market_cap`, or `adj_close`. Existing rows are
   preserved byte-for-byte, so higher-quality data already in the table (e.g.
   EDGAR market cap, better closes) is never lost.
3. is **resume-safe**: tickers that already have OHLC coverage are skipped, and
   re-runs are idempotent (fills are gap-only; new-date adds dedupe on
   (date, ticker)).

## Usage

```bash
python backfill_ohlcv.py                 # full universe (586 tickers)
python backfill_ohlcv.py --limit 20      # first 20 tickers (test)
python backfill_ohlcv.py --force         # refetch even tickers with OHLC
python backfill_ohlcv.py --delay 0.2     # faster (rate-limit friendly)
```

Run the full pass in the background (a few minutes to ~20 min depending on rate
limits).

## Data notes

- **Source column** is set to `yfinance` for fetched rows.
- **DATE-native:** the canonical `date` key is `datetime.date`; the script
  normalizes both the fetched rows and the saved parquet to it, so the sink
  stays date32[day].
- The `source` for rows fetched by an earlier (pre-fix) run may be NaN; re-run
  with `--force` for those tickers if provenance matters.

## Related

- `update_prices.py` — the incremental daily fetcher (close-first)
- `update_polygon.py` — the REST grouped-daily bulk fetch
- `statistical_profiler.py` — now able to use true VWAP once OHLCV is present
- `ride_longevity.py` — structural gate (ATR/chandelier) benefits from true OHLC
