# edgar_lib.py
Shared EDGAR library — frame parsing, fiscal year differencing, provenance tracking.

## Why it exists (rationale)

Multiple scripts (`backfill_edgar.py`, `edgar_companyfacts_v2.py`, `edgar_html_10q.py`, `unified_edgar_pipeline.py`, `acquisition_backfill.py`) need the same XBRL frame parsing, CIK resolution, quarterly differencing, and FCF computation logic. This module centralizes those operations so fixes (e.g., handling a new frame pattern, correcting fiscal YTD differencing) propagate to all consumers. It also defines the canonical `TAG_MAP` (XBRL tag lists for each financial concept) and `CIK_OVERRIDES` for shells/missing mappings.

## Usage

```python
from edgar_lib import (
    load_cik_map, get_cik, CIK_OVERRIDES, NO_COMPANYFACTS,
    extract_financials, compute_quarterly_fundamentals, detect_fiscal_year_end,
    parse_quarterly, parse_cashflow_quarterly, parse_balance, TAG_MAP,
    fetch_companyfacts, extract_facts
)
```

No CLI — this is a library module imported by other scripts.

## Outputs

- `.cik_cache.json` — cached SEC ticker→CIK map (regenerated on first run or cache miss)
- None otherwise — all functions return DataFrames/Series in memory.

## Related programs

- `edgar_companyfacts_v2.py` — primary consumer (enhanced parser)
- `edgar_html_10q.py` — uses CIK map and constants
- `unified_edgar_pipeline.py` — imports extraction and computation functions
- `backfill_edgar.py` — original consumer of shared EDGAR logic
- `acquisition_backfill.py` — uses CIK resolution for M&A detection