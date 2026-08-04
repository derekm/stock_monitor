# buy_candidates.py

Decision layer beyond the dual-pass gates: ranks names expected to rise and
emits a BUY / ACCUMULATE / WATCH / AVOID call with reasons.

## Why it exists (rationale)

`preferred_metrics` / `inclusion_criteria` say *whether* a name passes screens.
This script turns that into an actionable buy list by combining the screen
decision with momentum, the factor composite, regime posture (stress → higher
bar / fewer names), a liquidity floor, and leverage flags — and S&P 500
membership for liquidity/benchmark relevance.

## Usage

```bash
python buy_candidates.py --save
```

Flags: `--save` (write outputs). Reads `preferred_metrics.csv`,
`momentum_metrics.csv`, `factor_panel.csv`, `risk_metrics_ext.csv`,
`hmm_regimes.csv`, `sp500_sleeve.csv`.

## Outputs

- `buy_candidates.csv` — full ranked list with decision + reasons
- `buy_candidates_top.csv` — top candidates only

(Schema family: screen_decision — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [preferred_metrics.md](preferred_metrics.md) — screen source
- [factor_panel.md](factor_panel.md) / [momentum_analytics.md](momentum_analytics.md)
- [risk_metrics_ext.md](risk_metrics_ext.md)
- [inclusion_criteria.md](inclusion_criteria.md)
- [hmm_regime_detection.md](hmm_regime_detection.md)
