# buy_candidates.py

buy_candidates.py — Decision layer beyond dual-pass gates for names expected to rise.

## Why it exists (rationale)

Decision layer above the dual-pass gates: ranks names expected to appreciate using factor/momentum/risk signals, producing a tradable buy shortlist.

## Usage

```bash
python buy_candidates.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Screen / decision** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `buy_candidates.csv`
  - `buy_candidates_top.csv`
  - `factor_panel.csv`
  - `momentum_metrics.csv`
  - `risk_metrics_ext.csv`
- **Regime / state table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `hmm_regimes.csv`
- **Summary / metrics** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `preferred_metrics.csv`


## Related programs

- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/factor_panel.md](factor_panel.md)
- [docs/momentum_analytics.md](momentum_analytics.md)
- [docs/dual_screen_analysis.md](dual_screen_analysis.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
