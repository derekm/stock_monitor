# dual_screen_analysis.py

Explains *why* the Buffett-quality + value-trifecta dual pass is rare, and lists
external (not-yet-monitored) candidates that could approach it.

## Why it exists (rationale)

The dual pass (high ROE/ROIC + cheap valuation) is structurally rare: quality
names tend to be expensive, cheap names tend to be cyclical/low-quality. This
script quantifies the tension on the monitored book (the gap) and surfaces
external tickers worth monitoring — a research aid, not a screen output.

## Usage

```bash
python dual_screen_analysis.py --save
```

Flags: `--save` (writes the two CSVs).

## Outputs

- `dual_screen_gap.csv` — monitored names: quality vs value tension by ticker
- `dual_screen_external_candidates.csv` — external tickers that could approach
  dual pass (illustrative — verify with live data)

(Schema family: screen_decision — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [preferred_metrics.md](preferred_metrics.md) — the dual pass it analyzes
- [inclusion_criteria.md](inclusion_criteria.md) — gate definition
- [manage_stocks.md](manage_stocks.md) — add external candidates
- [dupont_analysis.md](dupont_analysis.md)
