# edgar_companyfacts_v2.py

Enhanced SEC EDGAR companyfacts parser with quarterly differencing, FCF proxy, and M&A data extraction.

## Why it exists (rationale)

The original `backfill_edgar.py` extraction was too strict — it required specific XBRL frame formats and bailed out entirely if any core series was missing. This module adds quarterly differencing for cumulative cash flow frames, FCF proxy when CapEx is unavailable, and M&A tag extraction for acquisition detection.

## Usage

```bash
python edgar_companyfacts_v2.py --tickers AAPL,MSFT  # extract specific tickers
python edgar_companyfacts_v2.py --max-tickers 10      # extract first 10 tickers
python edgar_companyfacts_v2.py --dry-run             # CIK coverage report only
```

## Outputs

- `edgar_v2_quarterly.csv` — extracted quarterly fundamentals (when run standalone)
- No direct parquet writes — designed to be imported by `backfill_edgar.py`

## Related programs

- `backfill_edgar.py` — calls this module's `extract_raw_financials()` and `compute_quarterly_fundamentals()`
- `edgar_lib.py` — shared EDGAR utilities (frame parsing, provenance)
- `unified_edgar_pipeline.py` — orchestrates v2 + HTML 10-Q fallback