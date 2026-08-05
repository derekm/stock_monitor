# run_daily_automation.py

Master daily job for the stock_monitor analytics stack — runs the full pipeline
in order as subprocesses.

## Why it exists (rationale)

After a data refresh, every analytics CSV must be regenerated in the right order
(screens before risk before export). This is the one command that does the whole
daily stack: it chains the screen → stress → risk → correlation → history →
dupont → growth → export jobs, each as a subprocess with a timeout, and reports
OK/FAIL/TIMEOUT per step.

## Job sequence

1. `preferred_metrics --save`
2. `inclusion_criteria --explore-defensive --save`
3. `stress_dual_pass --save`
4. `crisis_correlation --save`
5. `factor_rotation_defense --save`
6. `risk_enrich`
7. `rolling_window_analysis --universe portfolio --save`
8. `rolling_correlation_windows --save`
9. `tail_risk_hedging --save`
10. `allpairs_correlations --window 63 --step 21 --max-assets 50`
11. `fundamentals_history snapshot`
12. `fundamentals_history backtest-screens`
13. `dupont_analysis --save`
14. `growth_tech_analytics`
15. `export_dashboard_data`

## Usage

```bash
python run_daily_automation.py
python run_daily_automation.py --skip-growth --skip-allpairs
python run_daily_automation.py --only export,inclusion
```

Flags: `--skip-growth`, `--skip-allpairs`, `--only <comma list of job names>`.

## Outputs

None directly (each subprocess writes its own artifacts; see their docs).
Ends by exporting `dashboard_data/data.json`.

**Not in the job list (run manually):** `hmm_regime_detection.py` and
`rebalance_calendar.py` are **not** in `JOBS`. Run
`python hmm_regime_detection.py --save` then
`python rebalance_calendar.py --months 18 --save` after the daily stack if you want
the calendar artifact. `rebalance_calendar.csv` is not read by any other script (see
[KNOWN_ISSUES.md](KNOWN_ISSUES.md)).

## Related programs

- [maintain_analytics.md](maintain_analytics.md) — alternative "rebuild all" path
- [export_dashboard_data.md](export_dashboard_data.md) — final export step
- every program in the job list above
