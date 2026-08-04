# dupont_analysis.py

DuPont decomposition of ROE into profitability, efficiency, and leverage.

## Why it exists (rationale)

ROE alone is ambiguous: a high ROE can come from leverage, not quality. The
classic 3-step DuPont (Profit Margin × Asset Turnover × Equity Multiplier)
separates them so the Buffett lens — prefer margin/turnover-driven ROE, not
leverage — can be applied. It feeds the quality scoring in `preferred_metrics`.

## Usage

```bash
python dupont_analysis.py
python dupont_analysis.py --min-roe 0.15 --save
```

Flags: `--min-roe` (filter, default 0.0), `--save`.

## Outputs

- `dupont_analysis.csv` — per-ticker ROE decomposition (margin, turnover, EM)

(Schema family: summary_metrics — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [preferred_metrics.md](preferred_metrics.md) — consumes quality inputs
- [dual_screen_analysis.md](dual_screen_analysis.md)
- [update_fundamentals.md](update_fundamentals.md) — fundamental source
