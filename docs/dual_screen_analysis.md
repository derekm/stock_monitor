# dual_screen_analysis.py

## Why no dual Buffett + trifecta passes

| Screen | Median in passers |
|--------|-------------------|
| Quality (ROE/ROIC ≥15%, D/E≤1) | P/B ≈ **6.5**, EV/EBITDA ≈ **16** |
| Value trifecta | ROE ≈ **8%**, ROIC ≈ **6%** |

High-quality compounders are **priced for quality**. Cheap trifecta names are often **cyclical or low-ROE**. Buffett’s real style is closer to *wonderful business at a fair price* than to a strict three-ratio value screen.

## External near-dual candidates (not monitored)

See `dual_screen_external_candidates.csv` — e.g. HPQ, VLO/PSX/MPC (cycle-dependent), SYF, NUE/STLD, FANG/DVN. **Verify live fundamentals** before acting.

Outputs: `dual_screen_gap.csv` (quality-vs-value gap table), `dual_screen_external_candidates.csv` (external near-dual names).

```bash
python dual_screen_analysis.py
```

## Related programs

- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/factor_panel.md](factor_panel.md)
- [docs/buy_candidates.md](buy_candidates.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
