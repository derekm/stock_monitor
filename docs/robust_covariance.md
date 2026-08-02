# robust_covariance.py

Robust covariance estimators for ERC / GMV / Black-Litterman inputs.

| Estimator | Role |
|-----------|------|
| **sample** | Classical sample covariance |
| **ledoit_wolf** | Shrinkage toward scaled identity — stabilizes condition number |
| **oas** | Oracle Approximating Shrinkage |
| **ewma** | RiskMetrics-style λ=0.94 recursive cov |
| **winsorized** | Clip returns at ±2.5σ then sample cov |

```bash
python robust_covariance.py --universe portfolio --save
python robust_covariance.py --universe growth --window 126 --save
```

Outputs: `robust_covariance_summary.csv`, `cov_ledoit_wolf_{universe}.csv`
