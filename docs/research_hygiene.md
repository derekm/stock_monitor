# research_hygiene.py

Walk-forward inclusion rules + forecast reliability report.

## Why it exists (rationale)

Two hygiene checks for the research: (1) **walk-forward** — does the dual-pass
gate actually select winners out-of-sample? It backtests the inclusion rule
forward in time. (2) **forecast-reliability** — ranks setups by realized
accuracy. Together they keep the screen and forecast claims honest.

## Usage

```bash
python research_hygiene.py walk-forward --save
python research_hygiene.py forecast-reliability --save
python research_hygiene.py all --save
```

Sub-commands: `walk-forward`, `forecast-reliability`, `all`. Flag: `--save`.

## Outputs

- `walk_forward_inclusion.csv` — out-of-sample gate performance
- `forecast_reliability_report.csv` — setup reliability ranking

(Schema families: screen_decision / forecast_anomaly — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [inclusion_criteria.md](inclusion_criteria.md) / [dual_screen_analysis.md](dual_screen_analysis.md)
- [forecast_reliability.md](forecast_reliability.md)
- [fundamentals_history.md](fundamentals_history.md)
