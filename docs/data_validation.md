# data_validation.py
Guards against zero/negative/infinite prices, future dates, and malformed fundamentals.

## Why it exists (rationale)

Data ingested from SEC EDGAR, yfinance, and Polygon can contain sentinel values (zero/negative/infinite prices, future-dated rows, infinite fundamentals) that silently corrupt downstream factor calculations, backtests, and screens. This module provides the `validate_prices()` and `validate_fundamentals()` functions that strip or NaN-out these rows before they reach consumers like `cross_section.py` and `signal_aggregator.py`.

## Usage

```python
from data_validation import validate_prices, validate_fundamentals

clean_prices = validate_prices(raw_prices_df)
clean_fundamentals = validate_fundamentals(raw_fundamentals_df)
```

No CLI — this is a library module imported by other scripts.

## Outputs

- None — operates in-place on DataFrames passed to it. No files written.

## Related programs

- `data_integrity.py` — broader data integrity checks (price jumps, gaps)
- `data_integrity_deep.py` — deep integrity audit with reporting
- `backfill_edgar.py` — uses validation before writing fundamentals
- `update_prices.py` — uses validation before writing daily_prices