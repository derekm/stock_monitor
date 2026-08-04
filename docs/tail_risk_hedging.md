# tail_risk_hedging.py

tail_risk_hedging.py — Explore tail-risk hedging overlays for the defensive book.

## Why it exists (rationale)

Explores tail-risk hedging overlays (cash, low-vol, put-proxy, tail_combo) for the defensive book — informs `factor_rotation_defense`.

## Usage

```bash
python tail_risk_hedging.py [--index/--ticker/--sector/--save/--window ...]  # shared flags via cli_common
```

> Most programs accept the standard `cli_common` flags ([docs/cli_common.md](cli_common.md)): `--index`, `--ticker`, `--sector`, `--save`, `--window`, `--freq`. Check the script's `--help` for script-specific flags.


## Outputs

- **Index level series** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `daily_prices.parquet`
- **Base parquet table** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `fundamentals.parquet`
  - `monitored_stocks.parquet`
- **Other** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `tail_risk_hedge_crisis.csv`
- **Weights / performance** (see [docs/SCHEMAS.md](SCHEMAS.md)):
  - `tail_risk_hedge_performance.csv`


## Related programs

- [docs/factor_rotation_defense.md](factor_rotation_defense.md)
- [docs/monte_carlo.md](monte_carlo.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
