# risk_metrics_ext.py

Liquidity, concentration, and factor-style risk metrics (Polars + pandas).

## Why it exists (rationale)

Beyond vol/beta, the book needs liquidity and concentration risk visible. This
computes per-ticker liquidity + simple factor scores (`risk_metrics_ext.csv`)
and a portfolio-level concentration / liquidity / beta summary
(`portfolio_risk_summary.csv`) — the risk picture that informs caps and sleeve
sizing.

## Usage

```bash
python risk_metrics_ext.py --save
```

Flags: `--save`. Reads `daily_prices.parquet`, `portfolio_holdings.parquet`.

## Outputs

- `risk_metrics_ext.csv` — per-ticker liquidity + factor scores
- `portfolio_risk_summary.csv` — concentration / liquidity / beta summary

(Schema families: summary_metrics / weights_performance — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [risk_enrich.md](risk_enrich.md) — vol/beta/DD enrichment
- [vol_target.md](vol_target.md) / [risk_parity_analytics.md](risk_parity_analytics.md)
- [preferred_metrics.md](preferred_metrics.md)
