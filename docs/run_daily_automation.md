# run_daily_automation.py

Master daily job for the stock_monitor analytics stack — runs the full pipeline
as a dependency DAG with multiprocessing (independent jobs run in parallel;
only real dependencies serialize). Universe is daily_prices.

## Why it exists (rationale)

After a data refresh, every analytics output must be regenerated in the right
order (screens before risk before export). This is the one command that does
the whole daily stack: it loads `daily_automation_dag.yaml`, topologically
sorts the jobs by their declared dependencies, executes them in wave order
with configurable parallelism, and reports OK/FAIL/TIMEOUT per step.

## DAG configuration

**Single source of truth:** `daily_automation_dag.yaml` (not hardcoded in
Python). The YAML defines every job's command, timeout, description, and
dependency set. The runner loads it strictly — no fallback to in-code
definitions.

## Job list (52 jobs as of 2026-08-22)

| Job | Command | Depends on |
|-----|---------|------------|
| `hmm` | `hmm_regime_detection.py --n-states 3 --save` | — |
| `market_cap` | `add_daily_marketcap.py` | — |
| `rebalance` | `rebalance_calendar.py --months 18 --save` | `hmm` |
| `preferred` | `preferred_metrics.py --save` | — |
| `implied_r` | `implied_r_screen.py --save` | `preferred` |
| `momentum` | `momentum_analytics.py --save` | `preferred` |
| `inclusion` | `inclusion_criteria.py --explore-defensive --save` | `preferred` |
| `stress` | `stress_dual_pass.py --save` | `preferred, inclusion` |
| `crisis` | `crisis_correlation.py --save` | — |
| `factor_rot` | `factor_rotation_defense.py run --save` | — |
| `risk_enrich` | `risk_enrich.py` | `preferred` |
| `rolling` | `rolling_window_analysis.py --universe all --save` | `risk_enrich` |
| `rolling_corr` | `rolling_correlation_windows.py --save` | `preferred, risk_enrich` |
| `tail_hedge` | `tail_risk_hedging.py --save` | `rolling, hmm` |
| `allpairs` | `allpairs_correlations.py --window 63 --step 21 --max-assets 80` | `preferred` |
| `fund_snap` | `fundamentals_history.py snapshot` | `backfill_new_tickers` |
| `screen_bt` | `fundamentals_history.py backtest-screens` | `preferred, inclusion` |
| `backfill_new_tickers` | `acquisition_backfill.py` (timeout 1800s) | `acq_backfill` |
| `dupont` | `dupont_analysis.py --save` | `preferred` |
| `growth` | `growth_tech_analytics.py` | `preferred, dupont` |
| `peer` | `peer_analytics.py --save` | `preferred` |
| `earnings` | `earnings_catalyst.py --save` | `growth, peer` |
| `pairs` | `pair_engine.py --save` | `peer, earnings` |
| `cross` | `cross_section.py --save` | `peer, earnings, pairs` |
| `aggregate` | `signal_aggregator.py --save` | `cross, earnings, pairs, peer, preferred` |
| `technical` | `technical_signals.py --save` | `aggregate` |
| `econ_cal` | `economic_calendar.py --save` | — |
| `est_rev` | `estimate_revisions.py --save` | — |
| `shadow` | `shadow_book.py --save` | `preferred, aggregate` |
| `damodaran` | `damodaran_quality.py --all` | `preferred` |
| `lookthrough` | `lookthrough_engine.py` | `backfill_new_tickers` |
| `acq_backfill` | `acquisition_backfill.py` | — |
| `taleb_tail` | `tail_index.py` | `preferred` |
| `taleb_gap` | `gap_risk.py` | `preferred` |
| `taleb_iv_skew` | `iv_skew.py --skip-existing` (timeout 600s) | `preferred` |
| `taleb_ergodic` | `ergodicity_ruin.py` | `taleb_tail` |
| `taleb_fragility` | `fragility_screen.py` | `taleb_tail, taleb_gap, taleb_iv_skew` |
| `taleb_minsky` | `macro_fragility.py --save` | `hmm, taleb_fragility` |
| `taleb_shock` | `macro_shock.py --save` | `hmm` |
| `taleb_sector_shock` | `macro_sector_shock.py --save` | `hmm` |
| `taleb_shock_ride` | `shock_ride.py --save` | `taleb_sector_shock` |
| `taleb_arista` | `arista.py --save` | — |
| `taleb_ride_now` | `ride_now.py --save` | — |
| `taleb_subindustry_regime` | `subindustry_regime.py --save` | `taleb_sector_shock` |
| `taleb_barbell` | `barbell_check.py` | `taleb_fragility, taleb_ergodic` |
| `taleb_optionality` | `hidden_optionality_audit.py` | `aggregate, preferred` |
| `polygon_prices` | `update_polygon.py --days 5 --save` (timeout 300s) | — |
| **`bogle_tmi`** | `build_bogle_funds.py --fund tmi --save --years 10` (timeout 300s) | `polygon_prices` |
| **`bogle_qmi`** | `build_bogle_funds.py --fund qmi --save --years 10` (timeout 300s) | `polygon_prices` |
| **`bogle_bpi`** | `build_bogle_funds.py --fund bpi --save --years 10` (timeout 300s) | `polygon_prices` |
| `export` | `export_dashboard_data.py` | *(all analytics + bogle jobs)* |

> **New in 2026-08-22:** Three Bogle-style index funds (`bogle_tmi`, `bogle_qmi`,
> `bogle_bpi`) added as daily jobs, depending on `polygon_prices` for fresh
> price data and feeding into `export` for dashboard exposure.

## Usage

```bash
# Full daily refresh (all 52 jobs in dependency order)
python run_daily_automation.py

# Selective run (job + its transitive deps)
python run_daily_automation.py --only bogle_tmi,bogle_qmi,bogle_bpi
python run_daily_automation.py --only inclusion,stress,export

# Skip specific jobs
python run_daily_automation.py --skip taleb_fragility,taleb_minsky

# List all valid jobs with their dependencies
python run_daily_automation.py --list

# Adjust parallelism (default 4 workers)
python run_daily_automation.py --max-workers 8
```

Valid job names (for `--only` / `--skip`): `hmm, market_cap, rebalance, preferred, implied_r, momentum, inclusion, stress, crisis, factor_rot, risk_enrich, rolling, rolling_corr, tail_hedge, allpairs, fund_snap, screen_bt, backfill_new_tickers, dupont, growth, peer, earnings, pairs, cross, aggregate, technical, econ_cal, est_rev, shadow, damodaran, lookthrough, acq_backfill, taleb_tail, taleb_gap, taleb_iv_skew, taleb_ergodic, taleb_fragility, taleb_minsky, taleb_shock, taleb_sector_shock, taleb_shock_ride, taleb_arista, taleb_ride_now, taleb_subindustry_regime, taleb_barbell, taleb_optionality, polygon_prices, bogle_tmi, bogle_qmi, bogle_bpi, export`.

## Outputs

None directly — each subprocess writes its own artifacts (see their docs).
The final `export` job rewrites `dashboard_data/data.json` with all tables.

## Execution model

1. **Load DAG** from `daily_automation_dag.yaml` (fails hard if YAML missing
   or PyYAML not installed — no silent fallback).
2. **Topological wave sort** using declared `dependencies`.
3. **Execute waves sequentially**; within each wave, run jobs in parallel
   up to `--max-workers`.
4. **Report per-job**: OK/FAIL/TIMEOUT with truncated stdout/stderr.
5. **Exit code**: 0 if all OK, 1 if any FAIL/TIMEOUT.

## Key design decisions

- **YAML is canonical** — no hardcoded fallback. If the YAML is missing or
  malformed, the runner crashes (visible failure > silent wrong behavior).
- **Parallelism by wave** — only true dependencies serialize. Independent
  jobs (e.g., `hmm`, `market_cap`, `crisis`, `factor_rot`) run concurrently.
- **Per-job timeouts** — defined in YAML; `None` = no timeout.
- **Subprocess isolation** — each job runs in its own Python process via
  `subprocess.run`, so crashes don't cascade and memory is released between
  jobs.
- **CWD = DATA_DIR** — all jobs run from `stock_monitor/` so relative paths
  in scripts resolve correctly.

## Related programs

- `daily_automation_dag.yaml` — the DAG definition (edit to add/change jobs)
- `export_dashboard_data.py` — final export step
- `build_bogle_funds.py` — Bogle-style index fund construction (new)
- Every program in the job list above (see their individual docs)

## Adding a new daily job

1. Add entry to `daily_automation_dag.yaml` under `jobs:` and `dependencies:`.
2. Register its output tables in `export_dashboard_data.py` TABLES catalog.
3. Run `python run_daily_automation.py --list` to verify it's recognized.
4. Test with `python run_daily_automation.py --only <new_job>`.