# factor_panel.py

Multi-factor scoring panel: value, quality, momentum, low-vol, and leverage flag,
combined into one composite.

## Why it exists (rationale)

`preferred_metrics` (quality+value) and `momentum_metrics` are separate. This
script joins them into a single per-ticker factor panel, normalizes each factor,
and builds an equal-risk-contribution-style rank composite so one ranked list
drives the buy/size decision.

## Usage

```bash
python factor_panel.py --save
```

Flags: `--save` (write outputs). Reads `preferred_metrics.csv`,
`momentum_metrics.csv`.

## Outputs

- `factor_panel.csv` — full factor panel + composite per ticker
- `factor_panel_top.csv` — top 25 by composite

(Schema family: screen_decision — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [preferred_metrics.md](preferred_metrics.md) / [momentum_analytics.md](momentum_analytics.md) — sources
- [buy_candidates.md](buy_candidates.md) — consumes the composite
- [factor_rotation_defense.md](factor_rotation_defense.md)
