# run_daily_automation.py

Master daily job:

```bash
python run_daily_automation.py
python run_daily_automation.py --skip-growth --skip-allpairs
python run_daily_automation.py --only export,inclusion,stress
```

Jobs: preferred → inclusion → stress → rolling → allpairs → fund snapshot →
screen backtest → dupont → growth → export dashboard JSON.

Also callable via `POST /run/all-daily` on analytics_service (subset).

## Related programs

- [docs/maintain_analytics.md](maintain_analytics.md)
- [docs/preferred_metrics.md](preferred_metrics.md)
- [docs/inclusion_criteria.md](inclusion_criteria.md)
- [docs/stress_dual_pass.md](stress_dual_pass.md)
- [docs/allpairs_correlations.md](allpairs_correlations.md)
- [docs/dupont_analysis.md](dupont_analysis.md)
- [docs/growth_tech_analytics.md](growth_tech_analytics.md)
- [docs/export_dashboard_data.md](export_dashboard_data.md)
- [docs/SCHEMAS.md](SCHEMAS.md) (output schemas)
