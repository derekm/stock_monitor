# unified_edgar_pipeline.py
Unified EDGAR extraction pipeline — combines XBRL + HTML parsing with smart fallback.

## Why it exists (rationale)

XBRL companyfacts is the fastest and most reliable source, but some filers have incomplete XBRL (missing FCF, proxy-only values) or no XBRL at all. This pipeline runs the XBRL path first (`edgar_lib.py`), then conditionally falls back to HTML 10-Q parsing (`edgar_html_10q.py`) when XBRL FCF is "proxy" or "unavailable". It merges results by date, preferring HTML-derived FCF when XBRL is weak, and computes a quality score (0–100) per ticker so consumers can filter low-confidence extractions.

## Usage

```bash
python unified_edgar_pipeline.py --ticker PANW           # single ticker
python unified_edgar_pipeline.py --max-tickers 50        # first 50 monitored tickers
python unified_edgar_pipeline.py --no-html               # XBRL only, skip HTML fallback
```

## Outputs

- stdout — prints per-ticker extraction stats (XBRL count, HTML count, merged count, quality score, FCF provenance distribution) and a sample of merged rows. No file output; designed for research and validation runs.

## Related programs

- `edgar_lib.py` — XBRL extraction (primary path)
- `edgar_html_10q.py` — HTML 10-Q parsing (fallback path)
- `edgar_companyfacts_v2.py` — enhanced XBRL parser (alternative to edgar_lib)
- `backfill_edgar.py` — original companyfacts writer (writes to fundamentals.parquet)