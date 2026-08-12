# update_polygon_flatfiles.py

Daily OHLCV ingest from **Massive.com Flat Files** (S3-compatible bulk feed),
key-gated. Complements the per-ticker REST path (`update_polygon.py`) with a
bulk download: **one gzipped CSV per trading day covering ALL U.S. equities**.

## Why it exists (rationale)

Closes the "integrate data sources: Polygon (production)" TODO using Massive's
flat-files S3 endpoint — the right tool for bulk historical data without
thousands of REST calls. Day aggregates live at
`us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz` (all US equities, one
file per trading day, available ~11am ET the next day, history back to 2003).

## Method

- **Credentials** are read from `massive_credentials.json` (gitignored) or
  `MASSIVE_ACCESS_KEY_ID` / `MASSIVE_SECRET_ACCESS_KEY` env vars. Without them
  the script prints setup and exits 0 (no crash in the automation).
- **Endpoint** `https://files.massive.com`, **bucket** `flatfiles`,
  SigV4-signed via boto3.
- For each of the last `--days` trading days, downloads the gzipped day CSV,
  normalizes columns to the price spine (`date, ticker, open, high, low, close,
  volume, source='polygon_flat', market_cap`), and appends to
  `daily_prices.parquet` (dedup on date+ticker, keep last).

## Access note (verified 2026-08-12)

Live probe: the S3 credentials **authenticate** — listing the bucket and the
`us_stocks_sip/day_aggs_v1/2026/08/` prefix returns all expected files. But
`GetObject`/`download_fileobj` on those objects returns **403 Forbidden**,
which is a **plan-level download restriction** (day aggregates are included in
Starter+ plans, not the free Basic tier). A wrong secret would fail listing
too; since listing succeeds, the key/signing is correct and the block is
authorization. When the subscription includes flat-file downloads, re-run
`python update_polygon_flatfiles.py --days 5 --save` with no code change.

## Usage

```bash
# set creds (json file or env vars), then:
python update_polygon_flatfiles.py --days 5 --save
```

## Outputs

- Appends to `daily_prices.parquet` (base table, `source='polygon_flat'`).

## Related programs

- `update_polygon.py` — the per-ticker REST bulk path (grouped endpoint)
- `daily_prices.parquet` — the shared price spine
- `run_daily_automation.py` — registered as `polygon_flatfiles` job
