# update_fundamentals.py

Maintain P/B, market cap, total assets, and mktcap/assets ratios in
`fundamentals.parquet`.

## Why it exists (rationale)

The value-trifecta screen needs current P/B, market cap, total assets, and
mktcap/assets — but yfinance coverage is uneven. This lets you inspect the
current fundamentals (`show`) and enter/refresh them manually (`manual`) so the
trifecta legs stay current for the monitored book. Append-only by `as_of_date`.

## Usage

```bash
python update_fundamentals.py show [--ticker CF]
python update_fundamentals.py manual --ticker CF --market-cap-b 18.2 --total-assets-b 13.8 --pb 2.9
```

Sub-commands: `show`, `manual`. `manual` flags: `--ticker`, `--market-cap-b`,
`--total-assets-b`, `--pb`, and others (see source).

## Outputs

- `fundamentals.parquet` — appended/updated rows (by `as_of_date`)

(Schema family: base_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [preferred_metrics.md](preferred_metrics.md) — consumes the ratios
- [inclusion_criteria.md](inclusion_criteria.md) — trifecta legs
- [backfill_constituents.md](backfill_constituents.md) — bulk real fundamentals
- [dupont_analysis.md](dupont_analysis.md)
