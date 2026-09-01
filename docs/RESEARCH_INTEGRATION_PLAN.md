# Research Integration Plan: 22 Scholars → StockMonitor Stack

**Goal:** Systematically integrate each researcher's core contributions into our codebase, with measurable upgrades to specific files. Priority deep dives (1–7) first; remaining 15 in maintenance/extension cycles.

---

## Phase 1: Priority Deep Dives (Weeks 1–14)

### 1. Fama/French + Novy-Marx — Factor Construction Validation
**Status:** **Closed (2026-08-23)** — Gate ∩ NM **60/95 = 63%** (bar 80% fail). Residual IC **+0.0117** (bar +0.02 fail). Gate ≠ QMJ. Do not loosen 15/15/1.0.
**Target files:** `factor_library.py`, `preferred_metrics.py`, `signal_aggregator.py`
**Core papers:** FF 1993/2015, Novy-Marx 2013/2014
**Deliverables:**
- [x] Dated `total_assets` (323,150 rows; 8,992 names ≥2 dates) + filing AG
- [x] NM panels on filing calendar → `novymarx_*.parquet` (AG 8,983 names)
- [x] Gate vs NM write-up → `docs/QUALITY_GATE_COMPARISON.md`
- [x] `nm_quality` separate from `buffett_pass`; AG clipped ±100% for ranks
- [x] Component D/E + `D/E ≥ 0` **persisted**: D/E n=5,581, `buffett_pass` **98**, Gate ∩ NM **60/95 = 63%** (bar 80% fail)
- [x] QMI = NM `nm_score` top quintile (≥2 legs); dual-pass `value_pass` = trifecta **or** B/M ≥ median **or** EY ≥ median
- [x] `ff5_factors.parquet` MKT **−6.44%/16% vol** (corr TMI 0.78). RMW **+7.4%/11% vol** via Rev/A fallback (`gross_profit` column is empty). TMI remains the market book.
- [x] `factor_attribution.py` named to actual factor columns
- [x] CAPM residual IC (PIT ER_{t-1} vs r_t − β̂_{36m} MKT_t, **fixed MKT**): **+0.0117** / 85m (bar +0.02 — **fail**)

**Success metric (measured):** Gate ∩ NM-quality = **63%** after persisted D/E (bar 80% fail). CAPM residual IC on **fixed MKT** = **+0.0117** (bar +0.02 fail). Do not loosen 15/15/1.0.

**Parked (later — not gate-loosening):**
1. Backfill `gross_profit` then GP/A — **10k fill running** (`gp_fill.py` / edgar v2). Rebuild NM GP/A + true MKT after it exits.
2. Residual IC ≥ +0.02 after MKT is sane (PIT `daily_mcap.parquet`, not last-shares)
3. **Derived panels, not live `daily_prices` writes** (Windows lock):
   - [x] `daily_mcap.parquet` — PIT shares × adj_close, stock-only, `--save` does not touch `daily_prices`
   - [x] As-of share join (merge_asof backward) — TSM last mcap $2.16T
   - [x] ER/TMI/FF5 read `daily_mcap.parquet`. Do not write `market_cap` onto `daily_prices`.
   - [x] HMC FY26 from 6-K (JPY): NI −¥423.9B, equity ¥12.15T, ROE −3.5%. `html_20f` rank 110. v2 filled FY21–FY25. BAYRY AR25 already merged.
   - [x] Implied-r: NaN unless ROE>0 and P/B>0 (HPQ/CAG garbage gated)
   - [x] Ride on the book → `ride_book.parquet`. Scores 0.34–0.49 (none extend). SMCI 0.34 worst.
   - [x] DAG/ownership mcap readers: implied_r, portfolio_report, export panel, ownership network, Damodaran cross-holdings — all `daily_mcap`, not `daily_prices.market_cap`.
   - **Migrate derived attributes to panels? Yes.** `daily_prices` stays OHLCV. Writers never `os.replace` the price file for a derived column.

---

### 2. Ilmanen + Ang — Expected Return Framework
**Status:** **Active (2026-08-23)** — 4-pillar ER written; CF/DR + regime premia landed. OOS hit-edge **+6.3pp** pass; top−EW **+1.4%** fail.
**Target files:** `expected_returns.py`, `implied_r_screen.py`, `factor_library.py`
**Core papers:** Ilmanen *Expected Returns* (2011), Ang *Asset Management* (2014), Cochrane (2011)
**Deliverables:**
- [x] Ilmanen 4-pillar ranks → `expected_returns_decomp.parquet` (mcap = daily else shares×px; ER requires ≥2 pillars)
- [x] CF vs DR: `cf_yield = ROE / P/B`, `discount_rate = 2·ROE/(P/B+1)` → `implied_r_decomp.parquet` (496 names)
- [x] Ang regime-conditional FF means → `regime_factor_premia.parquet` (`factor_library.py --regime-premia`)
- [x] Carry into `macro_fragility.py` (`equity_carry` = median ER carry by quarter)
- [x] `expected_return_report` deferred; ER eligibility + panel mcap wired
- [x] OOS direction: 224m, drop |ret|>50%; hit-edge **+6.3pp** (pass); top−EW **+1.4%** (fail +5% return)

**Measured now:** Regime premia **recomputed on fixed FF5** (223d): MKT low_vol −0.4% / stress **+68%** / normal **−51%** (n=128/48/47). SMB still wild in low_vol. Do not trade regime sleeves. Extra SMB/MOM winsor is 1.2 hygiene. CF>DR 61/496.

---

### 3. Asness/Pedersen — Signal Aggregation + Cost-Aware Weighting
**Status:** **Active (2026-08-23)** — live IC + dyn weights landed. Sharpe dyn−static **−0.14** (bar +0.15 fail). Do not size on `--dynamic`.
**Target files:** `signal_aggregator.py`, `cost_model.py`
**Deliverables:**
- [x] Pedersen `w ∝ max(IC,0) / (turnover × cost) × decay × regime-conf` → `signal_weights_dynamic.parquet`
- [x] Family half-lives → `signal_decay_params.json` (preferred 126 / peer 63 / cross 21 / pair 10 / earnings 5)
- [x] Cost-aware QP → `optimal_signal_weights.parquet` (`signal_aggregator.py --qp`)
- [x] Dynamic composite on stored scores → `shadow_dynamic.parquet` (not a full paper book)
- [x] Dynamic Sharpe − static Sharpe: **−0.14** (1.41 vs 1.55, 225m, net 10bps — bar +0.15 **fail**). Family snapshots are not PIT; test used dated ER pillars + Pedersen HLs.

**Measured (live IC, 2026-08-20, regime=low_vol, after uncapped pairs):** IC pref 0.048 / peer **0.177** / pair **0.127** / cross 0.025 / earn 0.022. Dyn w: peer 60% / pref 36% / pair 2% / cross 2%. Pair IC now live; HL=10d keeps dyn weight tiny. Sharpe dyn−static still **−0.14**. Do not size on `--dynamic`.

---

### 4. Taleb/Spitznagel/Haghani — Hardened Taleb Layer
**Status:** **Started (2026-08-23)**
**Deliverables:**
- [x] Bias-corrected Hill + k-stability → `tail_index_robust.parquet` (`tail_index.py`)
- [x] Fragility veto (Hill α<2) → `fragility_veto.parquet` (SMCI raw α=1.98)
- [x] 90/10 TMI/BPI barbell: maxDD ratio **0.98** (bar <0.50 — fail; BPI is not long-vol)
- [x] Hidden optionality v2: copy of live audit → `optionality_audit_v2.parquet` (max flip **0.57%** on momentum)
- [x] Vince leverage space (TMI/BPI grid; max at f_tmi=1.50 / f_bpi=0)
**Target files:** `tail_index.py`, `fragility_screen.py`, `barbell_check.py`, `ergodicity_ruin.py`, `hidden_optionality_audit.py`, `buy_candidates.py`
**Core papers:** Taleb *Statistical Consequences of Fat Tails* (2020), Spitznagel *Safe Haven* (2020), Haghani & White *The Missing Billionaires* (2023)
**Deliverables:**
- [ ] **Tail index (Hill estimator)**: Upgrade `tail_index.py` with Taleb's bias-corrected Hill (α < 2 detection) + subsampling stability → `tail_index_robust.parquet`
- [ ] **Fragility veto**: Formalize veto as `P(ruin) > ε` where ruin = drawdown > 50% from peak, using Haghani's ergodicity math → `fragility_veto.parquet` (ticker × date × veto_flag)
- [ ] **Barbell construction**: Implement Spitznagel's "dedicated tail hedge" (not correlation diversification):
  - Core: 90% `bogle_tmi` (or portfolio)
  - Hedge: 10% OTM puts / VIX calls / long vol (model via `macro_shock.py`)
  - Rebalance quarterly with glide → `barbell_portfolio.parquet`
- [ ] **Hidden optionality audit**: Extend `hidden_optionality_audit.py` with American-option decision flip rates under stress (Taleb's "decision convexity") → `optionality_audit_v2.parquet`
- [ ] **Kelly/leverage space**: Replace simple Kelly in `kelly.py` with Vince's **Leverage Space** (multi-asset, path-dependent optimal f) → `leverage_space_sizing.parquet`

**Success metric:** Barbell portfolio max DD < 50% of core portfolio in 2020/2022 crises; veto reduces blowup frequency by ≥50%

---

### 5. López de Prado — ML Regime Work Upgrade
**Status:** **Measured (2026-08-25)** — CPCV − random **+0.1pp** (bar +3% fail). meta_y=0. Regime clustering **passes (+46.4% full-universe)**. Hybrid `peer_group` wired into consumers.
**Deliverables:**
- [x] Triple-barrier on book+CORE → `triple_barrier_labels.parquet`
- [x] Meta-labeling: **meta_y mean 0** (no name ride≥0.5)
- [x] CPCV on TMI lag features → `cv_splits.parquet`: CPCV acc **53.7%** vs random KFold **53.6%** (**+0.1pp**, bar +3% **fail**)
- [x] Feature-coef stability across 15 CPCV folds → `feature_stability.parquet` (lag1 0.37; ma21 sign-unstable)
- [x] Regime clustering (HRP + distance corr) → `regime_clustering.py` → `regime_clusters.parquet`, `regime_cluster_dispersion.parquet`, `regime_cluster_sweep.parquet`
- [x] **Clustered sectors** — named by dominant sector composition. `cluster_name` + hybrid `peer_group` columns.
- [x] **Hybrid peer_group wired into `peer_analytics.py` and `cross_section.py`** — tighter peer groups (financial_services_76, mixed_healthcare, energy_89) with GICS fallback for the 2% where clustering is looser.
- [ ] SHAP (tree SHAP; coef stability is the stand-in)

**Regime clustering — measured (2026-08-25), 4,678 liquid listed names (61% coverage), k = 11 = #GICS sectors:**

Bar: within-cluster pairwise-correlation dispersion ≥20% below the GICS-sector baseline.

| linkage | 3y | 5y |
|---|---|---|
| ward | **+28.1% PASS** | **+20.3% PASS** |
| average | **+24.6% PASS** | **+20.0% PASS** (exactly on the bar) |
| complete | **+22.3% PASS** | +18.7% **FAIL** |
| single | +12.4% **FAIL** | +7.3% **FAIL** |

**5/8 configs clear the bar; range +7.3% to +28.1%.** The headline default (average/5y) lands at **exactly +20.0%**, so this is a **fragile pass, not a robust one** — it is linkage-dependent, and single linkage (chaining) fails outright. Report the config with the number.

**Full-universe (ward/5y, 4,678 names): dispersion reduction +46.4%** (from 0.1326 to 0.0711) — the ≥20% bar clears comfortably on the full tape.

**Distance correlation beats Pearson** on a controlled comparison (same 150 names, 3y, ward — only the metric differs): **dcor +29.3% vs corr +23.0% (+6.3pp)**, i.e. the non-linear codependence López de Prado argues for is doing real work, not just re-deriving sector labels.

**The clusters are economically real, and finer than GICS.** Ward/5y splits what GICS calls one "Healthcare" sector into **drug distributors (MCK/COR/CAH)**, **managed care (UNH/CVS/ELV/CI/HUM)** and **pharma (LLY/JNJ/PFE/MRK/ABBV)** — three 100%-pure clusters. It also finds **gaming (EA/TTWO)** and **clean energy (ENPH/FSLR/PLUG)**, which GICS scatters across Communication Services / Technology / Utilities. That is the usable output: a better peer/basket grouping for `peer_analytics` and `cross_section`.

**Caveat that limits the headline number:** the size distribution is **very unbalanced** — one cluster holds **286 of 400 names (72%)**, with the rest as small satellites (sizes 1–49). So part of the dispersion "win" is achieved by peeling off a few tight niches while leaving a large heterogeneous core, not by partitioning the market evenly. The niches are genuinely useful; the aggregate ≥20% figure flatters the method. Balanced-partition variants (k larger, or cutting the dendrogram by inconsistency rather than `maxclust`) are the honest next step before using `cluster` as a drop-in replacement for `sector`.

**Does NOT replace the HMM.** The plan says "replace HMM in `hmm_regime_detection.py`", but the HMM labels **dates** by market features (mkt_ret/vol21/avg_corr) and feeds `pass6`/`pass8`/`regime_serving`; HRP clusters **assets** by codependence. They are different objects, and the ≥20%-dispersion metric is an asset-grouping metric. The clustering is therefore an **addition** (a better peer//basket grouping than GICS), and the HMM date labeller stays.
**Success metric (measured):** CPCV does **not** beat shuffled KFold by 3pp on this persistence task. Do not claim CPCV as a lift. Regime clustering **does** clear ≥20% on 5/8 configs (best dcor/ward **+29.3%**), but not on all.
**Target files:** `subindustry_regime.py`, `peer_analytics.py`, `cross_section.py`, `signal_model.py`, `hmm_regime_detection.py`, `regime_clustering.py`
**Core papers:** López de Prado *Advances in Financial ML* (2018): CPCV, meta-labeling, regime clustering, triple-barrier
**Deliverables:**
- [ ] **Meta-labeling**: Wrap `signal_model.py` GradientBoosting with meta-label (primary model = direction, meta = position size) → `meta_labeled_signals.parquet`
- [ ] **CPCV (Combinatorial Purged Cross-Validation)**: Replace random train/test in `signal_model.py` with CPCV (no leakage, respects time structure) → `cv_splits.json` + updated model
- [ ] **Regime clustering**: Replace HMM in `hmm_regime_detection.py` with López de Prado's **Hierarchical Risk Parity + regime clustering** (codependence + distance correlation) → `regime_clusters.parquet`
- [ ] **Triple-barrier labeling** for `peer_analytics.py` / `cross_section.py`: label each ticker-window with (touch upper, touch lower, timeout) → `triple_barrier_labels.parquet`
- [ ] **Feature importance stability**: Add SHAP stability across CPCV folds → `feature_stability.parquet`

**Success metric:** CPCV OOS accuracy > random-split OOS by ≥3%; regime clusters reduce within-cluster correlation dispersion by ≥20%

---

### 6. Hoffstein/Vince — Sequence Risk + Leverage Space
**Status:** **Started (2026-08-23)**
- [x] Rebalance luck: TMI 41q, median std **1.68%** → `rebalance_luck_distribution.parquet`
- [x] Vince 2-asset grid TMI/BPI: max at **f_tmi=1.50, f_bpi=0** (no hedge) → `leverage_space_allocation.parquet`
- [x] Optimal glide: **7-day** vs 1-day luck std **−50.8%** (bar 40% **pass**) → `optimal_glide_schedule.parquet`
- [x] CDaR / sequence risk on TMI: CDaR5 **−25.1%**, seq_risk **0.013**
- [x] Multi-period Kelly TMI: f **2.96** vs single **3.46** → `multi_period_kelly.parquet`
- [x] Vince LS vs ERC (400 block-bootstrap paths): LS median terminal **18.54** vs ERC **6.40** → `ls_vs_erc.parquet` (**LS dominates**)
**Success metric (measured):** 7-day glide **−50.8%** (bar 40% **pass**). LS **beats** ERC on median terminal. Do not size live books at f=1.50.
**Target files:** `rebalance_calendar.py`, `vol_target.py`, `kelly.py`, `portfolio_optimization.py`, `risk_parity_analytics.py`
**Core papers:** Hoffstein "Rebalancing Luck" (2019), "Sequence Risk" (2020), Vince *Leverage Space Trading Model* (2009), *The Leverage Space Model* (2013)
**Deliverables:**
- [ ] **Rebalancing luck quantification**: Monte Carlo `rebalance_calendar.py` over all possible rebalance days in quarter → `rebalance_luck_distribution.parquet`
- [x] **7-day linear glide** is the Hoffstein pick (luck −50.8%; 10-day better luck, worse day-0). Writers: `glide_rebalance(..., n_days=7)`.
- [ ] **Sequence risk metrics**: Add to `perf_metrics.py` — **sequence risk score** (correlation of returns with withdrawal phase), **conditional drawdown at risk** (CDaR)
- [ ] **Leverage Space sizing**: Implement Vince's multi-asset optimal f (joint distribution of returns, not marginal Kelly) → `leverage_space_allocation.parquet`
- [ ] **Path-dependent Kelly**: Replace `kelly.py` single-period with multi-period Kelly (accounting for volatility drag) → `multi_period_kelly.parquet`

**Success metric:** Glide path reduces rebalancing luck std by ≥40%; Leverage Space allocation dominates ERC risk parity in Monte Carlo

---

### 7. Lo/Amodei — Adaptive Markets + LLM Forecasting
**Status:** **Started (2026-08-23)**, adaptive-HMM claim **re-measured on full history (2026-08-24)**
- [x] Regime population shares (11m HMM file) → `regime_population.parquet`
- [x] Adaptive persist vs vol → `adaptive_hmm_states.parquet` (bar **+0.60 fail**)
- [x] Split conformal on TMI |r|/σ₂₁: coverage **88.9%** vs 90% bar → `conformal_bands.parquet` (**fail**, 1.1pp short)
- [x] LLM forecasting (prototype, 2026-08-30): `forecast_llm.py` local Llama-3.2-3B Instruct Q4, Python-owned brief, JSON grammar. Not fine-tuned. Not production.
- [ ] Ensemble weights

**Adaptive HMM (persistence vs vol) — CORRECTED 2026-08-24.** The recorded **−0.90 (n=169)** is not reproducible as a *regime-persistence* result. Measuring persistence the natural way — **regime run length vs mean in-run `vol21`** — gives essentially **zero** relationship on both samples:

| sample | corr | n runs | mean run len | dates | span |
|---|---|---|---|---|---|
| **full history** | **−0.074** | 278 | 57.9d | 16,087 | 1962-01-31 → 2026-08-07 |
| adaptive window | −0.075 | 15 | 15.4d | 231 | 2025-10-06 → 2026-08-24 |

Both **fail the +0.60 bar**, and both agree, so this is **not** a truncated-window artifact — the −0.90 came from a **different estimator** (a per-date persistence proxy over n=169 dates, not run lengths). Two estimators disagreeing by 0.83 means the metric definition was doing the work, not the market.

**Honest statement: regime persistence is ~uncorrelated with volatility (−0.07 over 64 years and 278 runs).** Do not quote −0.90, and do not claim "high vol shortens dwell" — the long sample does not support a sign in either direction. `adaptive_hmm_states.parquet` now stores **both samples with `n_runs`** so the estimator and sample size travel with the number.
**Success metric (measured):** run-length persistence/vol correlation **−0.074 (278 runs, full history)** vs bar +0.60 → **fail**. Conformal **88.9% < 90%** → fail.
**Target files:** `hmm_regime_detection.py`, `statistical_profiler.py`, `forecast_granite.py`, `granite_daily.py`, `regime_calibrate.py`, `regime_serving.py`
**Core papers:** Lo *Adaptive Markets Hypothesis* (2004/2017), Amodei et al. *Constitutional AI* (2022), Granite TTM papers (IBM 2023-2024)
**Deliverables:**
- [ ] **Adaptive HMM**: Extend `hmm_regime_detection.py` with Lo's **time-varying transition probabilities** (regime persistence changes with volatility) → `adaptive_hmm_states.parquet`
- [ ] **Population dynamics**: Add `statistical_profiler.py` metric: **regime population fitness** (fraction of tickers in each regime, evolution over time) → `regime_population.parquet`
- [x] **LLM forecasting integration**: `forecast_llm.py` — Llama-3.2-3B Instruct Q4 on MX550; Python writes the dossier; model writes a two-sentence JSON forecast. Coverage-gated set is **365**. Profiles in one long table: `value` (too-expensive → not up), `exuberant` (crowd can keep paying up), `compounder` (leftover cash + ROIC−WACC beat can carry expensive). Last HMM date only; one-shot; `n_ctx=1024`. Not FinGPT/BloombergGPT; not fine-tuned. Conformal still blocked on a 1-date series.
- [ ] **Uncertainty calibration**: Upgrade `regime_calibrate.py` with **conformal prediction** (distribution-free prediction intervals) + Amodei's **constitutional uncertainty** (model expresses "I don't know") → `conformal_bands.parquet`
- [ ] **Regime-selected ensemble**: Enhance `regime_serving.py` with **dynamic model weighting** (Lo's evolutionary weight update based on recent regime performance) → `ensemble_weights.parquet`

**Success metric:** Adaptive HMM regime persistence correlation with realized vol > 0.6; conformal bands achieve 90% coverage

---

## Phase 2: Core Extensions (Weeks 15–26)

**Spec (2026-08-30).** Phase 1 is not a clean pass: Gate ∩ NM **63%** (bar 80 fail), persist/vol **−0.074** (bar +0.60 fail), barbell maxDD ratio **0.98** (bar <0.50 fail), Sharpe dyn−static **−0.14**. Gate 3 is **not** closed. Phase 2 does not wait on a passing barbell, and does not reopen those bars.

Hard constraints from Phase 1, still in force:

- One researcher at a time. Order below.
- Sidecar panels. Never write derived columns onto `daily_prices`.
- Do not loosen Novy-Marx 15/15/1.0.
- Do not size on `--dynamic` (Sharpe −0.14).
- Do not put sector/macro tape or 0–1 aggregator scores in the LLM brief.
- Universe = `daily_prices/`. `monitored_stocks.parquet` is sleeve metadata.
- Verify by running on the full tape, not a 16-name toy.

**Already on disk (do not rebuild as Phase 2):** 12-1 in `momentum_analytics.py` / `expected_returns.py` / `factor_library.py`; TSMOM 3/6/12 + JT-6 in `momentum_research.py` (2026-08-11: hit-on ~0.60–0.67, spread ~+6–7.5%/yr); CF/DR in `implied_r_decomp.parquet` (CF>DR **0.7%** of rows — not a discriminator); dual-pass in `preferred_metrics.py`; 8-K lexicon in `filings_sentiment.py`.

**Blocked until coverage jobs finish:** universe SI (`companyfacts_cache/` is CRM only); `gross_profit` empty so GP/A cannot be Gray/Vogel QV; `sector_prices` levels are unusable (do not feed Faber GTAA).

### 8. Jegadeesh/Titman — Momentum Foundations (start here)

**Status:** Spec. 12-1 and JT-6 exist; **12-2** (skip last two months) is the missing paper replica.

**Target:** `momentum_analytics.py`, `fractal_windows.py`, `backtest_price_vs_momentum.py`, `momentum_research.py`

**Checklist:**
- [ ] Add `mom_12_2` column to `momentum_analytics.py` (252d return, skip last 42 trading days; PIT, no lookahead)
- [ ] Compute `mom_12_1`, `mom_12_2`, `mom_fractal` on identical date grids (same universe, same calendar)
- [ ] Long-short backtest (top/bottom quintile, equal weight, monthly rebalance, 10 bps cost) for each momentum definition
- [ ] Compare annualized spread: 12-2 vs 12-1 vs fractal on overlapping dates (full history from `daily_prices/`)
- [ ] Document fractal `momentum_stack` incremental contribution after costs vs 12-2
- [ ] If fractal does not beat 12-2 by +2 pp/yr net: reclassify fractal as ride tool, not JT replica in `docs/momentum_analytics.md`
- [ ] Write `momentum_jt.parquet` (ticker × date × `mom_12_1`, `mom_12_2`, `mom_fractal`)
- [ ] Fold results into `docs/momentum_analytics.md` (methodology + results table)

**Do not:** Re-implement TSMOM (that is item 13, already measured). Do not put 12-1 in the `value` brief when expensive (already omitted).

**Bar:** 12-2 long-short net of 10 bps beats 12-1 by **+2 pp annualized** over the overlapping tape, or fractal beats both by that amount. Else fractal is a ride tool, not a JT replica.

**Output:** `momentum_jt.parquet` (ticker × date × `mom_12_1`, `mom_12_2`, `mom_fractal`); `docs/momentum_analytics.md` fold-in.

### 9. Gray/Vogel — Quantitative Value/Momentum

**Status:** Blocked on GP/A. Dual-pass (ROE/ROIC + trifecta) is not Alpha Architect QV.

**Target:** `preferred_metrics.py`, `inclusion_criteria.py`, `dual_screen_analysis.py`

**Checklist:**
- [ ] Wait for `gross_profit` backfill (`gp_fill.py` / edgar v2) to complete — verify ≥8,000 names have GP/A
- [ ] Implement QV per paper: EV/EBITDA (cheap quintile) + GP/A (top quintile) + low leverage (D/E ≤ median) — all PIT on filing calendar
- [ ] Implement QM per paper: 12-1 momentum + `nm_quality` (≥2 legs: GP/A top quintile, low accruals, safe leverage)
- [ ] A/B test: QV vs current `value_pass` (trifecta ∨ B/M ∨ EY); QM vs `nm_score` top quintile
- [ ] Long-short backtest both against TMI with same costs (10 bps) and same calendar
- [ ] Report net spread, not just overlap percentages
- [ ] Write `gray_vogel_qv.parquet`, `gray_vogel_qm.parquet` (ticker × date × qv_flag, qm_flag, components)

**Do not:** Substitute Rev/A for GP/A and call it QV. Do not loosen 15/15/1.0 to make overlap.

**Bar:** QV∩NM ≥ **80%** on names with GP/A, or report fail. QV long-short vs dual-pass: quote net spread, not overlap theater.

**Output:** `gray_vogel_qv.parquet`, `gray_vogel_qm.parquet`.

### 10. Faber — GTAA + Shareholder Yield

**Status:** Not started. `sector_prices` levels are junk (Health Care −0.000123, Industrials 181,957). `build_bogle_funds.py` is equity indexes, not GTAA.

**Checklist:**
- [ ] Define asset-class universe from `daily_prices/`: map tickers → asset class (equity/bond/REIT/commodity/gold/USD) using a maintained mapping table
- [ ] Build equal-weight daily returns per asset class (PIT, no survivorship bias)
- [ ] Implement 10-month SMA trend per class (trend = price > SMA200; signal = 1/0; monthly rebalance)
- [ ] Backtest GTAA: trend-following allocation vs TMI (equal-weight equity) on full history
- [ ] Compute shareholder yield from fundamentals panel: `dividend_yield + net_buyback_yield` when both columns exist; fallback to dividend yield only
- [ ] Validate shareholder yield coverage ≥ 500 names on latest date
- [ ] Write `gtaa_sleeve.parquet` (date × asset_class × trend_flag, weight); `shareholder_yield.parquet` (ticker × date × sy)

**Do not:** Trend `sector_prices` levels. Do not put GTAA state in the LLM brief.

**Bar:** GTAA vs TMI: maxDD ratio **< 0.70** (2020 and 2022) and full-sample net CAGR within **2 pp** of TMI. Else it is a DD tool, not a return engine.

**Output:** `gtaa_sleeve.parquet`, `shareholder_yield.parquet`.

### 11. Cochrane — Discount Rate Decomposition

**Status:** Formula live. Coverage is the gap. `implied_r_screen` is **470** names; CF>DR **0.7%**.

**Checklist:**
- [ ] Keep `r = 2·ROE/(P/B+1)` gated ROE>0, P/B>0, r>0 (`4e0d1ca` commit)
- [ ] Expand screen: compute implied-r for every name with valid ROE, P/B, price on each filing date
- [ ] Verify latest implied-r notna count ≥ 2,000 names with r>0
- [ ] If < 2,000: diagnose binding hole (ROE≤0 count, P/B≤0 count, price missing count) and document in `docs/implied_r_coverage.md`
- [ ] CF vs DR: compute `cf_yield = ROE / P/B`, `discount_rate = 2·ROE/(P/B+1)`; report top/bottom tercile in consumers
- [ ] Do not dump 0–1 scores; word as "CF>DR" / "CF<DR" / "inconclusive" in briefs
- [ ] Write updated `implied_r_screen.parquet` / `implied_r_decomp.parquet` (same schema, longer coverage)

**Do not:** Link CF/DR to `macro_shock` as a second cost-of-capital line. Do not treat CF>DR as a signal until it is not 0.7%.

**Bar:** Latest implied-r notna **≥ 2,000** names with r>0, or report the binding hole (ROE vs P/B vs price).

**Output:** same `implied_r_screen.parquet` / `implied_r_decomp.parquet` (long, not a new schema).

### 12. Baker/Wurgler — Sentiment + Catering

**Status:** Lexicon MVP only (`filings_sentiment.py`). Not a market sentiment index.

**Checklist:**
- [ ] Extract dated 8-K sentiment per ticker: filing text → lexicon score (residual = score − 21d market return)
- [ ] Build panel: `filings_sentiment_residual.parquet` (ticker × filing_date × residual)
- [ ] Catering test: high residual (top tercile) → subsequent issuance (SEO/debt) / buyback in next 63 trading days
- [ ] Test IC of residual vs next-21d residual return using CPCV (López de Prado splits from `cv_splits.parquet`)
- [ ] If IC ≤ 0 on CPCV: drop catering claim, keep only lexicon as brief sidecar
- [ ] Keep Polygon firehose mentions (`ticker_news_mentions.parquet`) separate — they feed brief press line, not this index

**Do not:** Mix Polygon firehose tags into 8-K sentiment. Do not HMM-interact until the residual has IC.

**Bar:** 8-K residual IC vs next-21d residual return **> 0** on a CPCV split, or drop the catering claim.

**Output:** `filings_sentiment_residual.parquet`.

### 13. Moskowitz — Time-Series Momentum

**Status:** Absorbed by item 8. TSMOM 3/6/12 already measured (~+7%/yr spread, hit ~0.60).

**Checklist:**
- [ ] Only run after item 8 bar passes (12-2 vs fractal decision is made)
- [ ] Implement vol-scaled TSMOM-12: `sign(r_12) / σ_20` as date-level overlay (same universe, same calendar as item 8)
- [ ] Backtest vs CS 12-2 on identical dates, same costs (10 bps)
- [ ] Compare Sharpe ratios; if vol-scaled TSMOM-12 Sharpe ≤ CS 12-2 Sharpe: keep CS only, write one-line fail note
- [ ] If bar passes: write `tsmom_overlay.parquet` (date × ticker × tsmom_signal, weight)

**Do (only after 8's bar):** Vol-scale TSMOM-12 (`sign(r_12) / σ_20`) as a **date-level** overlay vs CS 12-2, same costs.

**Bar:** Vol-scaled TSMOM-12 Sharpe **> CS 12-2 Sharpe** on the same tape, or keep CS only.

**Output:** `tsmom_overlay.parquet` if the bar is tested; else a one-line fail in `docs/momentum_research.md`.

### 14. Perold/Sharpe — CPPI + Risk Parity

**Status:** ERC exists (`risk_parity_analytics.py`). Vince LS already **beats** ERC on median terminal (18.54 vs 6.40). CPPI floor+multiplier is new.

**Checklist:**
- [ ] Implement CPPI on TMI: floor = 90% peak NAV, multipliers m ∈ {2,3,4}
- [ ] Simulate 400 block-bootstrap paths (same paths as Vince LS vs ERC test in `ls_vs_erc.parquet`)
- [ ] Compare: CPPI (m=2,3,4) vs ERC vs HRP vs Vince LS on median terminal wealth, maxDD, drawdown duration
- [ ] Test across multiple start dates (not just one crisis)
- [ ] Document cost assumptions (rebalancing frequency, transaction costs)

**Do not:** Size live books at Vince f=1.50. Do not declare CPPI a winner on one crisis.

**Bar:** CPPI maxDD **< ERC maxDD** and terminal wealth **not** worse than ERC median by 20%. Else ERC stays the risk-parity claim; LS stays the leverage claim.

**Output:** `cppi_paths.parquet`.

### 15. Merton — ICAPM + Multi-Hedge

**Status:** Last. `black_litterman.py` exists. Macro panels are stale or junk (ERP already inside WACC; `exogenous_panel` ~1 month behind prices).

**Checklist:**
- [ ] Build hedge sleeves from liquid instruments: TIPS (bond proxy), USD (currency proxy), labor-income proxy (use SPY sector-neutral construction per Santos & Veronesi 2006)
- [ ] Implement ICAPM hedging: estimate hedge coefficients from Merton's multi-factor model on rolling 252d windows
- [ ] Backtest: ICAPM-hedged TMI vs TMI (net of hedge costs) on full history
- [ ] BL views: only from Phase 1 signals that passed bars (ER hit-edge +6.3pp ✓; dyn weights ✗)
- [ ] Write `icapm_hedge_sleeves.parquet` (date × sleeve × weight)

**Do not:** Put inflation/oil/CPI in `forecast_llm.py`. Do not invent a labor-income factor from GICS.

**Bar:** ICAPM-hedged TMI vs TMI: maxDD ratio **< 0.85** with net CAGR loss **< 1.5 pp/yr**. Else BL stays a view engine, not an ICAPM claim.

**Output:** `icapm_hedge_sleeves.parquet`.

**Phase 2 sequence:** 8 (12-2 vs fractal) → 11 (implied-r coverage, unblocked) → 13 only if 8 passes → 9 when GP/A exists → 10 after asset-class returns exist → 14 → 12 → 15. Dashboard tile per closed item. Shadow-book costs before any live sleeve.

---

## Phase 3: Specialized Layers (Weeks 27–38)

### 16. Spitznagel — Tail Hedging Deepening
**Target:** `tail_risk_hedging.py`, `macro_shock.py`, `barbell_check.py`
**Deliverable:** Dedicated tail hedge construction (SPX put ladder, VIX calls, long vol ETFs); cost/drag analysis

### 17. Haghani — Ergodicity + Missing Billionaires
**Target:** `ergodicity_ruin.py`, `kelly.py`, `leverage_space_sizing.parquet`
**Deliverable:** Full ergodicity economics calculator; household-level ruin probability

### 18. Vince — Leverage Space Complete
**Target:** `kelly.py`, `vol_target.py`, `portfolio_optimization.py`
**Deliverable:** Full Leverage Space optimizer (joint return surface, genetic algorithm)

### 19. Amodei/Karpathy — LLM Forecasting Production
**Target:** `forecast_granite.py`, `granite_daily.py`, `regime_serving.py`
**Deliverable:** Production LLM forecaster (fine-tuned on our data + macro); constitutional uncertainty

### 20. Asness (Managed Futures) — Crisis Alpha
**Target:** `tail_risk_hedging.py`, `macro_shock.py`, `factor_rotation_defense.py`
**Deliverable:** Managed futures factor (trend + carry across futures); crisis alpha attribution

### 21. Ang (Alternative Risk Premia) — Factor Zoo
**Target:** `peer_analytics.py`, `cross_section.py`, `signal_aggregator.py`
**Deliverable:** Catalog of 50+ alternative risk premia; factor zoo dashboard

### 22. Novy-Marx (Profitability/Investment) — Quality Deepening
**Target:** `preferred_metrics.py`, `dupont_analysis.py`, `quality_scores.parquet`
**Deliverable:** Full Novy-Marx quality factor (gross profitability, investment, accruals); replace our ad-hoc gate

---

## Cross-Cutting Infrastructure (Continuous)

| Infrastructure | Target Files | Owner |
|----------------|--------------|-------|
| **Unified factor library** | `factor_library.py` (new) — single source for FF5, QMJ, carry, TSMOM, etc. | Phase 1 |
| **Experiment tracking** | `mlflow`/`wandb` integration for all model training (Granite, signal_model, HMM) | Phase 1 |
| **Data versioning** | DuckLake migration for `daily_prices`, `fundamentals`, `factor_library` | Phase 2 |
| **Dashboard tiles** | One tile per researcher framework (FF attribution, Ilmanen ER, Taleb fragility, etc.) | Each phase |
| **Documentation** | `docs/research_<name>.md` per researcher: theory → our implementation → validation | Each deliverable |

---

## Sequencing Rules

1. **No new files without validation** — every deliverable must have a backtest/verification script
2. **One researcher at a time** — complete deep dive before starting next (avoids partial integrations)
3. **Backward compatibility** — old scripts keep working; new outputs are additive parquet tables
4. **Dashboard first** — each deliverable must appear in `export_dashboard_data.py` TABLES before merge
5. **Cost honesty** — every strategy paper-traded in `shadow_book.py` with full cost model before live consideration

---

## Resource Allocation

| Role | Focus |
|------|-------|
| **Quant Engineer** (primary) | Phases 1–3 implementation, backtests, dashboard |
| **ML Engineer** | Phase 1.5 (López de Prado), Phase 2.7 (Lo/Amodei LLM) |
| **Data Engineer** | Factor library, DuckLake, data versioning, pipeline reliability |
| **Reviewer** (you) | Validate each deliverable against original paper; approve merge |

---

## Milestone Gates

| Gate | Criteria | Decision |
|------|----------|----------|
| **Gate 1 (Week 4)** | FF5 replication validated; quality gate comparison done | **Closed (fail bar).** Overlap **60%** (bar 80%). `buffett_pass` ≠ QMJ. Residual IC open. Do not loosen 15/15/1.0. |
| **Gate 2 (Week 8)** | Dynamic signal weighting beats static; expected return decomposition live | Continue Phase 1 |
| **Gate 3 (Week 14)** | Taleb layer hardened; barbell portfolio backtested | **Not closed.** Barbell maxDD ratio **0.98** (bar <0.50 fail). Phase 2 spec started anyway; barbell stays a Phase 1 leftover. |
| **Gate 4 (Week 26)** | Item 8 bar (12-2 vs 12-1 / fractal +2 pp) and item 11 (≥2,000 implied-r names) | Enter remaining Phase 2 (9–10, 14) |
| **Gate 5 (Week 38)** | LLM forecaster production; Leverage Space complete | Maintenance mode |

---

## Quick Start (This Week)

```bash
# 1. FF5 replication on our universe
python -c "
from src.analytics.factor_library import compute_ff5
ff5 = compute_ff5('daily_prices/')
ff5.to_parquet('ff5_factors.parquet')
print(ff5.tail())
"

# 2. Quality gate vs Novy-Marx
python -c "
import pandas as pd
fund = pd.read_parquet('fundamentals.parquet')
# Compute Novy-Marx quality: GP/A, low accruals, safe leverage
# Compare to our gate
"
```

---

*Plan version: 2026-08-30. Phase 2 spec: 12-2 first; do not rebuild TSMOM/CF-DR. Update after each gate review.*