# data_integrity_deep.py

Deeper price / fundamental integrity scan than `data_integrity.py`.

## Why it exists (rationale)

A second, more aggressive pass: multi-threshold jump scan, suspected split
factors (integer-ish price ratios), stale-quote / flat-line detection,
cross-sectional same-day outlier scores, a fundamental missingness report, and
alignment coverage across price ∩ membership ∩ preferred screens. Catches the
subtler corruption the first pass misses.

## Usage

```bash
python data_integrity_deep.py --save
```

Flags: `--save` (write the report CSVs/JSON).

## Outputs

- `data_integrity_deep_summary.json` — summary counts
- `suspected_splits.csv` — price-ratio split candidates
- `price_flatlines.csv` — stale / flat-line series
- `fundamental_missingness.csv` — per-ticker fundamental gaps
- `alignment_coverage.csv` — price ∩ membership ∩ preferred overlap

(Schema families: aux_table / summary_metrics — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [data_integrity.md](data_integrity.md) — first-line gate
- [preferred_metrics.md](preferred_metrics.md) — one of the alignment inputs
- [manage_stocks.md](manage_stocks.md) — membership source
