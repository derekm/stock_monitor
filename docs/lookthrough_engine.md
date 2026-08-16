# lookthrough_engine.py
Generalized Pro Forma Financial Combination for Acquisitions.

## Why it exists (rationale)

When a company acquires another, its reported fundamentals only include the target from the completion date forward. For quarters between announcement and the first combined report, consumers need pro forma combined numbers to avoid artificial jumps in revenue, margins, and leverage ratios. This engine reads `corporate_actions.parquet` to determine active look-through windows, then additively combines acquirer + target fundamentals for each quarter in the window. It tracks two provenance columns (`data_provenance`: standalone vs. lookthrough_proforma; `lookthrough_source`: which tickers were combined) so downstream analytics can filter or flag pro forma rows.

## Usage

```python
from lookthrough_engine import get_pro_forma_fundamentals, add_acquisition

# Get pro forma fundamentals (with look-through applied)
df = get_pro_forma_fundamentals("PANW", as_of_date="2026-06-30")

# Register a new acquisition
add_acquisition(
    acquirer_ticker="PANW",
    target_ticker="CYBR",
    completion_date="2026-07-31",
    announcement_date="2026-03-15",
    purchase_price=1_500_000_000,
)
```

No CLI — this is a library module. Running the module directly executes an example that registers PANW+CYBR and PANW+ZS acquisitions and prints the pro forma series.

## Outputs

- `corporate_actions.parquet` — acquisition records with look-through windows (schema family: other)
- Returns DataFrames in memory with `data_provenance` and `lookthrough_source` columns

## Related programs

- `acquisition_backfill.py` — detects acquisitions and calls `add_acquisition()`
- `fundamentals.parquet` — the base table being combined (schema family: base_table)
- `edgar_lib.py` — shared EDGAR utilities used for CIK resolution