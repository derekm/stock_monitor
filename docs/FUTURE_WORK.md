

### Better quant
- ~~Live fundamentals API (replace seeds/backfill noise)~~ **done** — yfinance `fetch-history` + SEC EDGAR XBRL backfill (9,340 rows, ~2006→2026)
- ~~Real crisis labels (2008, 2020, 2022) on long history~~ **thresholds found (2026-08)** — crisis labels derived from OUR data, not the papers. Using the market drawdown history (1962→2026, equal-weight index) as crisis ground truth and the `macro_fragility` signals read point-in-time (no lookahead) at each onset:
  - **Empirical threshold: `crisis_band` (debt impulse ≥ 0.20) preceded 6 of 7 US crises** (1987, 2000, 2008, 2020, 2022, 2026). The one miss (1973-74, impulse 0.162) was the pre-1980 low-debt era.
  - **Minsky-signal pctile ≥ 80% fired at 5 of 6 pre-2026 crises** (2008 was the extreme at 99th pctile, impulse 0.365).
  - **2011 eurozone is a genuine true negative**: impulse 0.087 (elevated, 20th pctile) — an external crisis our US-debt-driven signal correctly did not flag.
  - The `danger_zone` column already emits these labels daily; the **open** part is a permanent labeled-episode table (`crisis_labels.csv`: onset, trough, min-DD, signal readings) + a validation harness scoring the signal's crisis-detection hit rate / false-positive rate out-of-sample.
- ~~Options-based hedges (put spreads) vs put_proxy~~ **partial** — ATM IV + skew/put-call now tracked (`options_skew.py`); spread construction still open
- ~~Transaction-cost and tax-aware rebalancing~~ **done** — `cost_model.py` (10bps + borrow) threaded into pair/cross backtests; shadow book tracks FIFO lots
- Walk-forward dual-pass / rotation (no look-ahead) — **synergy: the crisis-label table gives walk-forward its validation epochs** (post-crisis holdout windows), and `danger_zone` can gate the dual-pass book into defensive mode the way the stress posterior already gates buys
- Shrinkage + Black–Litterman views from screens automatically
- Partial-duration or inflation factors for true multi-asset defense
- ~~Decision robustness to estimation noise~~ **done (2026-08)** — the hidden-optionality audit (American-options paper) and the soft-stress / noise-convolved fixes: buy decisions now read the HMM posterior and E[g(x+ε)] driver contributions; flip rates dropped from 28.4% (regime) / 6.8% (momentum) to ~1.6% / ~6% with the knife-edges gone
- **FF5 + Novy-Marx factor library** **done (2026-08)** — `factor_library.py` replicates FF5+MOM on our universe + Novy-Marx quality factors; `signal_factor_loadings.parquet` maps signal families to factor betas; `signal_aggregator.py --use-residuals` for factor-neutral scoring
- **Ilmanen 4-pillar expected returns** **in progress** — `expected_returns.py` implements carry/value/momentum/defensive decomposition; outputs `expected_returns_decomp.parquet`
- **Factor attribution** **done** — `factor_attribution.py` rolling FF5 regression on portfolio returns (bogle_tmi R²=0.61, alpha=16.8%, beta_MKT=0.1)
- Quality gate validation: our gate (ROE≥15%, ROIC≥15%, D/E≤1.0) aligns with Novy-Marx (63%ile profitability, 64%ile low accruals, 32%ile low leverage, 89%ile value) — 85 tickers pass full gate

### Deeper integration
- **Single “decision engine”** reading inclusion + risk + regime → target weights (closest current: `buy_candidates.py` merges HMM/RISK/AGG extras into the composite)
- Dashboard: one-click “proposed trades” vs holdings
- Unify Fisher, Granite forecast, and factor rotation on shared calendar
- Alerting when dual-pass set changes or crisis corr regime flips — **synergy: `danger_zone` band transitions are a natural alert** (e.g. `danger`→`crisis_band` fired 2025Q4; the last such transition preceded the 2026 drawdowns). A crisis-correlation monitor (rolling_correlation_windows already exists) fed by the same crisis labels would complete the loop.
- ~~Paper-trade ledger with slippage vs SPX/SPY~~ **done** — `shadow_book.py` (buy_candidates targets replayed against realized prices, FIFO lots, kill switches)

### Data integrity
- Price pipeline health checks (the independent-synthetic corr ≈ 0 failure mode)
- Factor-structured or vendor data as default for regime research
- **LLM-brief coverage gaps (come back):** see [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) §5 Data integrity. Snapshot 2026-08-30: SI CRM-only; implied-r 470 with 28 negatives; buy_candidates 552; ride_book 9; veto 13; Unclassified 53.4%; quality 0 on **4,161 / 9,345**; fair EV/EBITDA 486/9345 with 461/486 g=2%; coverage-gated desk-note set **365**. `mos_pass` now requires traded EV/EBITDA > 0. Sector/macro stay out. Stay one-shot. Do not LLM-write the brief.

### Architecture gaps (known, not yet built)

These are concrete holes in the current architecture, distinct from the research wish-list above:

- **Per-ticker live-data joins are not wired.** Several analytics were originally written to join against a single stand-in ticker and were later generalized to apply uniformly, but the *proper* join of each analytic back to every ticker's live data (prices, fundamentals, forecasts, screens) does not yet exist. Today each program re-reads the spine tables directly; there is no shared "join analytics to all tickers' current state" layer. This is the next integration step before a single decision engine.
- **No single decision engine.** Screens, risk, and regime each emit their own outputs; nothing yet reads inclusion + risk + regime together and emits one target-weight set. `portfolio_optimization` / `risk_parity_analytics` consume the bands manually. `buy_candidates.py` is the closest partial (composite + gates + soft stress + de-noised drivers), but the full loop is not closed. **Synergy: the macro layer (`macro_fragility` danger_zone / Minsky signal) is a fourth input class the decision engine should read** — a macro risk-budget gate in the same spirit as the soft-stress posterior, not another per-ticker driver.
- **DuckLake not yet adopted.** `fundamentals_history.py` keeps dated snapshots and `backfill_*` captures history, but the large time-series tables are still committed as full parquet snapshots (per the data-versioning decision record). The planned DuckLake catalog for versioned PIT history is not implemented.
- **Forecast → screen feedback is still mostly one-directional.** Regime-selected serving (pass6/7/8 → `regime_serving.py`) now makes the forecast *regime-aware* and the signal aggregation is consumed by `buy_candidates.py`, but forecast direction does not yet re-weight the screen bands or risk budgets directly — it annotates the dashboard overlay and the candidate list. The forecasting-paradox upgrades (`forecast_nu`, `--epistemic-error`) make the forecast *uncertainty* honest, but it still doesn't drive allocation.

---

### Research Integration Plan (Phase 1 Priority Deep Dives)

Source of truth: [RESEARCH_INTEGRATION_PLAN.md](RESEARCH_INTEGRATION_PLAN.md). Status is the measured gate.

| # | Researcher | Status | Measured |
|---|------------|--------|----------|
| 1 | Fama/French + Novy-Marx | Closed (2026-08-23) | Gate ∩ NM **63%** (bar 80% **fail**). Residual IC **+0.0117** (bar +0.02 **fail**). Do not loosen 15/15/1.0. |
| 2 | Ilmanen + Ang | Active | OOS hit-edge **+6.3pp pass**; top−EW **+1.4% fail**. Word ER ranks; never dump 0–1. |
| 3 | Asness/Pedersen | Active | Sharpe dyn−static **−0.14** (bar +0.15 **fail**). Do not size on `--dynamic`. |
| 4 | Taleb/Spitznagel/Haghani | Started | Veto 13 names (SMCI). Barbell maxDD ratio **0.98** (bar <0.50 **fail**). |
| 5 | López de Prado | Measured (2026-08-25) | CPCV **+0.1pp fail**. Clustering **+46.4% pass**. Does not replace HMM. |
| 6 | Hoffstein/Vince | Started | 7-day glide **−50.8% pass**. Do not size live at f=1.50. |
| 7 | Lo/Amodei | Started | Persist/vol **−0.074 fail**. Conformal **88.9% fail**. LLM desk note live; coverage-gated **365**. See [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) §5. |

---

