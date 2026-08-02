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
