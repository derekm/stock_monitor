# tspulse_anomaly.py

Anomaly detection with TSPulse model hook + robust statistical backend (return z, residual z, volume z, market dispersion shocks).

```bash
python tspulse_anomaly.py status
python tspulse_anomaly.py scan --index portfolio --z 2.5 --save
```

Output: `anomalies_tspulse.csv` (dashboard Anomalies tab + CSV Catalog).

## Related programs

- [docs/analyze_granite_forecasts.md](analyze_granite_forecasts.md)
- [docs/forecast_granite.md](forecast_granite.md)
- [docs/update_prices.md](update_prices.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
