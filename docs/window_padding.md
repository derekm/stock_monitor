# window_padding.py

Fill a sub-512 context for short-history tickers (Granite TTM has a fixed
512-token context).

## Why it exists (rationale)

New S&P additions (spinoffs, recent IPOs) often have < 512 trading days, but the
TTM-r2 context can't be shrunk. To give them a real (non-fabricated) 512-window,
this pads the **head** with a genuine market/sector proxy rescaled to the
ticker's own price level: `[proxy_rescaled (first 512−n), ticker_actual (last n)]`.
A library used by the backfill/forecast path so short-history names still get a
valid window.

## Key functions

- `pad_to_context(ticker, close, sector=None)` → 512-length padded context
- `needs_backfill(ticker, px=None)` → (bool, days_short) — whether padding is needed
- `_sector_proxy(...)` — picks the rescaled proxy series

## Outputs

None written (library; returns padded arrays). Used by `ttm_backfill` /
`granite_backfill` / `forecast_granite`.

## Related programs

- [ttm_backfill.md](ttm_backfill.md) / [granite_backfill.md](granite_backfill.md)
- [forecast_granite.md](forecast_granite.md)
- [sp_universe_tracking.md](sp_universe_tracking.md) — recent additions
