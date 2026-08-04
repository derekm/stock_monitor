# tspulse_anomaly.py

Anomaly detection for stock series (TSPulse-ready).

## Why it exists (rationale)

Bad prints, halts, and structural breaks corrupt signals. This runs
anomaly detection over each stock's series (IBM Granite TSPulse targets TS
anomaly detection/classification) to flag suspicious windows — feeding
data-quality review and the dashboard's anomaly view. Also consumes the exogenous
panel for context.

## Usage

```bash
python tspulse_anomaly.py --save
python tspulse_anomaly.py --universe portfolio
```

Flags (via `cli_common` + own): `--universe/--index`, `--ticker`, `--save`. Reads
`daily_prices.parquet`, `monitored_stocks.parquet`, `exogenous_panel.parquet`.

## Outputs

- `tspulse_anomalies.csv` — flagged anomaly windows per ticker
  (plus related anomaly tables written alongside)

(Schema family: forecast_anomaly — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [data_integrity.md](data_integrity.md) / [data_integrity_deep.md](data_integrity_deep.md)
- [ttm_exogenous.md](ttm_exogenous.md) — exogenous panel source
- [forecast_granite.md](forecast_granite.md)
