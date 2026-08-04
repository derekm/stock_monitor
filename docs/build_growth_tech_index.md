# build_growth_tech_index.py

Fourth index sleeve: **higher-risk growth / technology** equal-weight basket.

## Sleeves (`growth_sleeve`)

| Sleeve | Names | Role |
|--------|-------|------|
| **growth_ai** | SMCI, NVDA, AMD, PLTR, CRWD | AI hardware/software — high vol |
| **quality_growth** | MSFT, GOOGL | Quality growth on dips |
| **emerging_growth** | TSLA, ENPH, SEDG, REGN, XBI | EV, solar cyclical, biotech |
| **cyclical** | BA, CAT, SCHW | Recovery / levered cyclicals |
| **thematic** | ARKK, QQQ, VUG | Speculative / ETF satellite — **small slice** |

## Usage

```bash
python build_growth_tech_index.py
python forecast_granite.py forecast --index growth --horizon 10
python run_fisher_duckdb.py --universe growth --save
# API
curl "http://127.0.0.1:5055/forecast/index?name=growth&horizon=10"
```

## Risk framing

- Not a defensive sleeve — higher σ, higher drawdown risk.
- Prefer **fractional Kelly / hard portfolio caps** (especially SMCI, TSLA, ARKK).
- Pair with fertilizer + defensive + personal indexes for multi-sleeve allocation.
- New listings may use synthetic prices until live backfill.

## Outputs

- `growth_tech_index.parquet` — membership snapshot + weights
- `growth_tech_index_levels.parquet` — EW index path (base 100)
- Flag: `monitored_stocks.growth_tech_index`

## Volatility targeting (example)

Use `python vol_target.py --ticker SMCI --save` to enforce a **5% max weight** and inverse-vol sizing. See [vol_target.md](vol_target.md).


## Starlink supply chain & launch (added)

| Sleeve | Tickers |
|--------|---------|
| starlink_supply | MCHP, TTMI, MTSI, STM, LRCX, AMAT, INTC, SATS (EchoStar) |
| launch_services | RKLB, ASTS, BKSY, SPCX |
| maritime_launch | TDW, VAL (offshore/maritime proxies) |
