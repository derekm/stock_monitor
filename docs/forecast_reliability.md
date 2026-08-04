# forecast_reliability.py

forecast_reliability.py — Rank forecast setups on holdings after first trade.

## Why it exists (rationale)

Ranks forecast setups by post-first-trade reliability so `forecast_granite` effort is spent where it actually adds signal; feeds `research_hygiene`.

## Usage

```bash
python forecast_reliability.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Forecast / anomaly** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `forecast_backtest_metrics.csv`
  - `forecast_reliability_detail.csv`
  - `forecast_reliability_rank.csv`


## Related programs

- [docs/forecast_granite.md](forecast_granite.md)
- [docs/research_hygiene.md](research_hygiene.md)
- [docs/granite_daily.md](granite_daily.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
