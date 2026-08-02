# tspulse_anomaly.py

Anomaly detection with TSPulse model hook + robust statistical backend (return z, residual z, volume z, market dispersion shocks).

```bash
python tspulse_anomaly.py status
python tspulse_anomaly.py scan --index portfolio --z 2.5 --save
```

Output: `anomalies_tspulse.csv` (dashboard Anomalies tab + CSV Catalog).
