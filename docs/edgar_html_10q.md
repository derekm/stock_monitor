# edgar_html_10q.py

HTML 10-Q parser for SEC filings — extracts financial tables from HTML-formatted quarterly reports.

## Why it exists (rationale)

Some filers report financials only in HTML format without complete XBRL tagging. This parser extracts the income statement, balance sheet, and cash flow tables directly from the HTML, providing a fallback when XBRL companyfacts is incomplete.

## Usage

```bash
python edgar_html_10q.py --tickers AAPL,MSFT  # parse specific tickers
python edgar_html_10q.py --max-tickers 10      # parse first 10 tickers
```

## Outputs

- `edgar_html_10q.csv` — extracted quarterly fundamentals from HTML
- No direct parquet writes — designed to be imported by `unified_edgar_pipeline.py`

## Related programs

- `edgar_companyfacts_v2.py` — XBRL-based extraction (primary)
- `unified_edgar_pipeline.py` — orchestrates XBRL + HTML fallback
- `backfill_edgar.py` — calls the pipeline for fundamentals