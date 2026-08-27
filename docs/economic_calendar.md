# economic_calendar.py

Trading-day, options-expiry, FOMC and earnings-event calendar for the
monitored universe.

## Why it exists (rationale)

Closes the "economic calendars" TODO. Regime-aware scheduling and
earnings-adjacent analytics need to know when events land. Zero new
dependencies: trading days come from `daily_prices/` (the actual
market calendar the repo trades on), quarterly expiries are computed (3rd
Friday of Mar/Jun/Sep/Dec), FOMC meetings come from a curated schedule (the
Fed publishes meetings years ahead — update `macro_events.csv` or the
embedded default annually), and earnings dates come from
`earnings_calendar.parquet`.

## Method

- Trading days: distinct dates in the price spine (16,264 for this universe).
- Quarterly option expiries: 3rd Friday of Mar/Jun/Sep/Dec within the spine
  range ± 400 days.
- FOMC: `macro_events.csv` if present, else embedded 2025–2027 schedule
  (source `federalreserve.gov`).
- Earnings: all `earnings_calendar.parquet` rows within the last 7 days.
- `days_until` and `is_trading_day` flags per event.

## Usage

```bash
python economic_calendar.py --save
```

## Outputs

- `economic_calendar.csv` — (date, event_type, label, source, days_until,
  is_trading_day), sorted, deduped.

## Related programs

- `earnings_calendar.parquet` — earnings events source
- `rebalance_calendar.py` — the regime-driven rebalance dates this complements
