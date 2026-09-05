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
- [x] **Re-measured 2026-09-01 after `gross_profit` backfill (real GP/A, not Rev/A)**: Gate ∩ NM **169/368 = 45.9%** (bar 80% — fail, worse); gate grew 98→370 as fundamentals coverage widened; 177/370 gate names lack the GP leg. FF5 RMW **+9.50%/11.35% vol**, MKT **−2.59%/14.45%**. Residual IC **+0.0169** / 84m (bar +0.02 — still fail). 15/15/1.0 untouched.

**Success metric (measured):** Gate ∩ NM-quality = **63%** after persisted D/E (bar 80% fail). CAPM residual IC on **fixed MKT** = **+0.0117** (bar +0.02 fail). Do not loosen 15/15/1.0.

**Parked (later — not gate-loosening):**
1. ~~Backfill `gross_profit` then GP/A~~ — **fill landed (113,717 rows, 2,817 names). Rebuilt 2026-09-01:** `novymarx_gross_profitability.parquet` latest nn **2,810** (was Rev/A fallback); `ff5_factors.parquet` RMW **+9.50%/11.35% vol** (was +7.4% via Rev/A), MKT **−2.59%/14.45%** (was −6.44%/16% on the stale snapshot). Fixed `load_prices` / `residual_ic` hive-read for `daily_prices/` partition layout; repaired truncated `daily_prices` 08-21/08-24 (12.5k missing tickers refetched via `update_polygon.py`), which now **detects partial dates** (`_partial_dates`, <50% of universe) and re-fetches them instead of leaving gaps. `daily_mcap.parquet` dense through **2026-08-31** (5,355 names).
2. Residual IC after MKT sane: **+0.0169** (84m, fixed MKT) — bar +0.02, **still fail**, up from +0.0117. Re-measure after GP/A panel flows into ER.
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
**Status:** **Measured (2026-08-23; bars below)** — all five layers built; the barbell **fails** its bar (BPI is not long-vol). Phase 2 does not reopen these bars.
**Deliverables (all built + measured):**
- [x] Bias-corrected Hill + k-stability → `tail_index_robust.parquet` (`tail_index.py`) — SMCI raw α=1.98
- [x] Fragility veto (Hill α<2, P(ruin) proxy) → `fragility_veto.parquet`
- [x] Barbell: 90/10 TMI/BPI, quarterly glide → `barbell_portfolio.parquet` — maxDD ratio **0.98** (bar <0.50 — **fail**; BPI is not long-vol)
- [x] Hidden optionality v2 (decision flip rates) → `optionality_audit_v2.parquet` (max flip **0.57%** on momentum)
- [x] Vince leverage space (TMI/BPI grid) → `leverage_space_sizing.parquet` (max at f_tmi=1.50 / f_bpi=0)
**Target files:** `tail_index.py`, `fragility_screen.py`, `barbell_check.py`, `ergodicity_ruin.py`, `hidden_optionality_audit.py`, `buy_candidates.py`
**Core papers:** Taleb *Statistical Consequences of Fat Tails* (2020), Spitznagel *Safe Haven* (2020), Haghani & White *The Missing Billionaires* (2023)
**Success metric:** Barbell portfolio max DD < 50% of core portfolio in 2020/2022 crises; veto reduces blowup frequency by ≥50% — **barbell fails (0.98).**

---

### 5. López de Prado — ML Regime Work Upgrade
**Status:** **Measured (2026-08-25)** — CPCV − random **+0.1pp** (bar +3% fail). meta_y=0. Regime clustering **passes (+46.4% full-universe)**. Hybrid `peer_group` wired into consumers.
**Deliverables:**
- [x] Triple-barrier on book+CORE → `triple_barrier_labels.parquet`
- [x] Meta-labeling: **meta IC 0.152 vs primary IC 0.220 (delta −0.068, 2026-09-01 re-run)** — meta-labeling **hurts**; `meta_y` mean 0 (no name ride ≥0.5). Implemented, measured, **do not size on it**.
- [x] CPCV on TMI lag features → `cv_splits.parquet`: CPCV acc **53.7%** vs random KFold **53.6%** (**+0.1pp**, bar +3% **fail**)
- [x] Feature-coef stability across 15 CPCV folds → `feature_stability.parquet` (lag1 0.37; ma21 sign-unstable)
- [x] **SHAP stability across CPCV folds (2026-09-01)**: `signal_model.py --shap-stability` → `feature_stability.parquet` now carries `shap_mean/shap_std/stability`. Ranked: cross **3.39** > preferred **1.95** > peer **1.48** > pair **1.06** > earnings **0.54**. Cross-sectional family is the most stable input; earnings is the least (its SHAP magnitude is also ~1,000× smaller — weak, not unstable).
- [x] Regime clustering (HRP + distance corr) → `regime_clustering.py` → `regime_clusters.parquet`, `regime_cluster_dispersion.parquet`, `regime_cluster_sweep.parquet`
- [x] **Clustered sectors** — named by dominant sector composition. `cluster_name` + hybrid `peer_group` columns.
- [x] **Hybrid peer_group wired into `peer_analytics.py` and `cross_section.py`** — tighter peer groups (financial_services_76, mixed_healthcare, energy_89) with GICS fallback for the 2% where clustering is looser.
- [x] SHAP (tree SHAP) — `--shap-stability`; see deliverable list

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
**Deliverables (all built + measured):**
- [x] **SHAP (tree SHAP)** — `--shap-stability` live; coef stability no longer the stand-in
- [x] **Meta-labeling**: `signal_model.py --meta-label` — primary GBC for direction + meta GBR for size. **Measured: meta IC 0.152 vs primary 0.220 (delta −0.068) — no lift, do not size on it.** → `signal_model_meta_oos.parquet`, `signal_model_meta_weights.parquet`
- [x] **CPCV** — `cv_splits.parquet` via `cv_utils.cpcv_folds`; CPCV acc **53.7%** vs random **53.6%** (+0.1pp, bar +3% **fail**). Do not claim CPCV as a lift.
- [x] **Regime clustering** — `regime_clustering.py` (HRP + distance corr) → `regime_clusters.parquet`; full-universe dispersion −46.4% (ward/5y), 5/8 configs clear ≥20% bar. **Addition, not HMM replacement** (HMM labels dates; clustering groups assets).
- [x] **Triple-barrier labeling** → `triple_barrier_labels.parquet` (touch upper/lower/timeout on book+CORE)
- [x] **Feature importance stability** — SHAP + coef stability across CPCV folds → `feature_stability.parquet` (SHAP stability: cross 3.39 > preferred 1.95 > peer 1.48 > pair 1.06 > earnings 0.54)

**Success metric:** CPCV OOS accuracy > random-split OOS by ≥3% — **fail (+0.1pp)**; regime clusters reduce within-cluster correlation dispersion by ≥20% — **pass on 5/8 configs, best +29.3%.**

---

### 6. Hoffstein/Vince — Sequence Risk + Leverage Space
**Status:** **Measured (2026-08-23)** — glide **pass** (−50.8%), LS > ERC. Do not size live books at f=1.50.
**Deliverables (all built + measured):**
- [x] Rebalance luck: TMI 41q, median std **1.68%** → `rebalance_luck_distribution.parquet`
- [x] Vince 2-asset grid TMI/BPI: max at **f_tmi=1.50, f_bpi=0** (no hedge) → `leverage_space_allocation.parquet`
- [x] Optimal glide: **7-day** vs 1-day luck std **−50.8%** (bar 40% **pass**) → `optimal_glide_schedule.parquet`
- [x] CDaR / sequence risk on TMI: CDaR5 **−25.1%**, seq_risk **0.013** → `perf_metrics.py` (`cdar()`, `sequence_risk()`)
- [x] Multi-period Kelly TMI: f **2.96** vs single **3.46** → `multi_period_kelly.parquet`
- [x] Vince LS vs ERC (400 block-bootstrap paths): LS median terminal **18.54** vs ERC **6.40** → `ls_vs_erc.parquet` (**LS dominates**)
**Target files:** `rebalance_calendar.py`, `vol_target.py`, `kelly.py`, `portfolio_optimization.py`, `risk_parity_analytics.py`
**Core papers:** Hoffstein "Rebalancing Luck" (2019), "Sequence Risk" (2020), Vince *Leverage Space Trading Model* (2009), *The Leverage Space Model* (2013)
**Success metric:** Glide path reduces rebalancing luck std by ≥40% — **pass (−50.8%)**; Leverage Space allocation dominates ERC risk parity in Monte Carlo — **pass (18.54 vs 6.40).**

---

### 7. Lo/Amodei — Adaptive Markets + LLM Forecasting
**Status:** **Started (2026-08-23)**, adaptive-HMM claim **re-measured on full history (2026-08-24)**
- [x] Regime population shares (11m HMM file) → `regime_population.parquet`
- [x] Adaptive persist vs vol → `adaptive_hmm_states.parquet` (bar **+0.60 fail**)
- [x] Split conformal on TMI |r|/σ₂₁: coverage **88.9%** vs 90% bar → `conformal_bands.parquet` (**fail**, 1.1pp short)
- [x] LLM forecasting (prototype, 2026-08-30): `forecast_llm.py` local Llama-3.2-3B Instruct Q4, Python-owned brief, JSON grammar. Not fine-tuned. Not production.
- [x] Ensemble weights — spec'd; **blocked on pass8 RPT sweep** (GPU, multi-day; `regime_model_best_rpt.parquet` has 5 rows — AEP only)

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
- [x] **Adaptive HMM** — `adaptive_hmm_states.parquet` (both samples with `n_runs`); persistence/vol corr **−0.074** (278 runs, full history) vs bar +0.60 → **fail**. Do not quote −0.90.
- [x] **Population dynamics** — `regime_population.parquet` (regime population shares, 11m HMM file)
- [x] **LLM forecasting integration**: `forecast_llm.py` — Llama-3.2-3B Instruct Q4 on MX550; Python writes the dossier; model writes a two-sentence JSON forecast. Coverage-gated set is **365**. Profiles in one long table: `value` (too-expensive → not up), `exuberant` (crowd can keep paying up), `compounder` (leftover cash + ROIC−WACC beat can carry expensive). One-shot; `n_ctx=1024`. Not FinGPT/BloombergGPT; not fine-tuned.
- [ ] **Multi-date forecast timeseries (unblocks conformal)**: `forecast_llm.py` accepts `--dates-file` / `--as-of` and writes one long parquet keyed `(date, ticker, profile)` — latest date keeps full snapshots, historical dates are PIT-only (undated snapshot panels like preferred/momentum/fragility are dropped to avoid lookahead). Resume key is `(date, ticker, profile)`. **11-date run in progress (2026-09-01, 365×3×11, resumable);** `conformal_bands.py` scores y = 1 if forward market return over the row's own `horizon_days` > 0 and applies per-regime 90% split-conformal. Bar 0.90 coverage — not yet measured.
- [ ] **Uncertainty calibration**: conformal bands above + Amodei's **constitutional uncertainty** (model expresses "I don't know"; `uncertainty_flag` already emits high/normal) → `conformal_bands.parquet`
- [ ] **Regime-selected ensemble**: Enhance `regime_serving.py` with **dynamic model weighting** (Lo's evolutionary weight update based on recent regime performance) → `ensemble_weights.parquet` — blocked on pass8 RPT sweep (GPU, multi-day)

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

**Status:** **Measured (2026-09-01) — bar FAIL on full tape.** 12-2 long-short beats 12-1 by **+1.3 pp/yr net** (12-1 **+16.4%**, 12-2 **+17.8%**, 448 months, 10 bps/side) — directionally right (longer skip helps), **below the +2 pp bar → FAIL**. All four 12-month fractals lose to the JT pair: b3_21 +11.0%, b3_42 +11.8%, b6_21 +10.9%, b6_42 **+12.1%** (best fractal − best JT = **−5.6 pp → FAIL**). **Fractal remains a ride tool, NOT a JT replica.** b6 (2-month bars, finer granularity) edges b3 at skip-42 (+0.3 pp) — granularity helps slightly but not decisively. Item 13 stays locked (rule: `13 only if 8 passes`).

**Target:** `momentum_analytics.py`, `fractal_windows.py`, `backtest_price_vs_momentum.py`, `momentum_research.py`

**Checklist:**
- [x] Add `mom_12_2` column (252d return, skip last 42 trading days; PIT, no lookahead) — computed in `momentum_research_backtest.py --jt`
- [x] Compute `mom_12_1`, `mom_12_2`, `mom_fractal` on identical date grids (same universe, same calendar)
- [x] 12-month fractal variants: **b3 ladder** `(21,3)(42,3)(63,3)(84,3)` → full windows 63/126/189/252d and **b6 fine view** `(42,6)` (2-month bars × 6) — both with JT skip parity (21d = 12-1 analog, 42d = 12-2 analog), computed PIT from the full tape in `_fractal_stack_series()`
- [x] Long-short backtest (top/bottom quintile, equal weight, monthly rebalance, 10 bps cost) for each momentum definition
- [x] Compare annualized spread: 12-2 vs 12-1 vs fractals on overlapping dates (**448-month full tape**)
- [x] Document fractal `momentum_stack` incremental contribution after costs vs 12-2 — **negative (best fractal +12.1% vs 12-2 +17.8%)**
- [x] **Fractal = ride tool, NOT JT replica** (−5.6 pp vs best JT) — folded into `docs/momentum_analytics.md`
- [x] Write `momentum_jt.parquet` (annualized LS table: net/gross/n_overlap per signal)
- [x] Fold results into `docs/momentum_analytics.md` (methodology + results table)

**Do not:** Re-implement TSMOM (that is item 13, locked: `13 only if 8 passes` — item 8 FAILED on full tape). Do not put 12-1 in the `value` brief when expensive (already omitted).

**Bar:** 12-2 long-short net of 10 bps beats 12-1 by **+2 pp annualized** over the overlapping tape, or fractal beats both by that amount. Else fractal is a ride tool, not a JT replica. → **12-2 beats 12-1 by +1.3 pp (FAIL); no fractal clears best JT (FAIL).**

**Output:** `momentum_jt.parquet` (date × net/gross LS return per signal); `docs/momentum_analytics.md` fold-in.

### 9. Gray/Vogel — Quantitative Value/Momentum

**Status:** **Re-measured (2026-09-01) after EV/EBITDA data-gap fix — QV∩NM bar PASS, QV still thin, QM FAILS.** `dual_screen_analysis.py --gray-vogel`, monthly-rebalance EW long-short vs TMI, 10 bps/side, PIT ffill on filing calendar.

**Data-gap fix (this session):** stored `ev_ebitda` was filled by an `ebit + capex` proxy over only 1,879 names / 870 at peak. Now computed from components (`EV = mcap + debt − cash`, real `ebitda` column): **2,588 names at latest date** (4.3× coverage, back to ~2018 with real EBITDA). Canonical fill LANDED 2026-09-02 once the LLM writer exited: `compute_missing_metrics.py` on real `ebitda` raised stored `ev_ebitda` 22,761 → **70,176 rows**; re-rerun of `--gray-vogel` on the filled panel reproduces the same verdict (QV +24.0%/yr, QV∩NM 87.9% PASS, QM −4.0%/yr FAIL) — stable.

**Results (re-measured):**
- **QV now fires on 30–36 names/date** on the EV-data-rich tape (was max 31 ever, median 0); **113 dates with ≥10 QV names**; still median 0 pre-2018 (no real EBITDA history).
- **QV∩NM = 87.9%** on names with GP/A (bar ≥ 80% — **PASS**, now on a non-vacuous 30+ name long leg).
- **QV net +24.0%/yr** (139 months with ≥20-name long legs, 10 bps) — improved and now a real book on recent tape, but pre-2018 EV/EBITDA still missing so the full backtest is recent-span only.
- **QM (12-1 momentum + nm_quality ≥2 legs): real book, LOSES: −4.0%/yr net, −3.9% vs TMI.** Unchanged — quality+momentum underperforms on this tape.
- Remaining gap: EBITDA history before ~2018 (EDGAR companyfacts backfill scope); GP/A is fine (2,817).

**Target:** `preferred_metrics.py`, `inclusion_criteria.py`, `dual_screen_analysis.py`

**Checklist:**
- [x] Wait for `gross_profit` backfill — done (113,717 rows, 2,817 names GP/A; ≥8,000 not met, Rev/A NOT used)
- [x] Implement QV per paper: EV/EBITDA (cheap quintile) + GP/A (top quintile) + low leverage (D/E ≤ median) — PIT ffill on filing calendar (`dual_screen_analysis.py --gray-vogel`)
- [x] Implement QM per paper: 12-1 momentum + `nm_quality` (≥2 legs: GP/A top quintile, low accruals, safe leverage)
- [x] A/B test: QV vs current `value_pass`; QM vs `nm_score` top quintile — **QV∩NM 93.3% (vacuous — QV fires on 0 names/date median); QM loses**
- [x] Long-short backtest both against TMI with same costs (10 bps) and same calendar — **QM −4.0%/yr net, −3.9% vs TMI**
- [x] Report net spread, not just overlap percentages — QV's +22.5% is a 0–31-name long leg, **not quoted**
- [x] Write `gray_vogel_ls.parquet` (annualized LS + QV∩NM); note: full ticker×date × qv/qm flag panel not written — do not build on an empty-flag signal

**Do not:** Substitute Rev/A for GP/A and call it QV. Do not loosen 15/15/1.0 to make overlap.

**Bar:** QV∩NM ≥ **80%** on names with GP/A, or report fail. QV long-short vs dual-pass: quote net spread, not overlap theater.

**Output:** `gray_vogel_qv.parquet`, `gray_vogel_qm.parquet`.

### 10. Faber — GTAA + Shareholder Yield

**Status:** **Measured (2026-09-02) — DD-tool bars PASS, CAGR bar FAIL → GTAA is a drawdown tool, not a return engine. Shareholder-yield coverage FAILS the 500-name bar (287 of 365 coverage names pay dividends).** `gtaa_trend.py`: real asset-class ETFs from the price hive (SPY/IWM, EFA/VWO, VNQ/RWR, AGG/LQD/TIP/HYG, GLD/SLV/DBC/GSG/DBA, BIL cash), class index = EW member returns, Faber 10-month SMA, monthly rebalance, weights lagged one month. Same-window 2016-09→2026-08 (TMI starts 2016): GTAA **CAGR +4.3%, vol 6.6%, Sharpe 0.07** vs TMI **CAGR +23.8%, vol 17.0%, Sharpe 1.11** (TMI is survivorship-gated to today's large caps — the bar's TMI is inflated, but even so GTAA's absolute CAGR is far below its equity benchmark). Crisis windows: maxDD ratio **0.30 (2020), 0.17 (2022)** — both under the 0.70 bar. Latest sleeve 2026-08-31: US equity, intl equity, REITs, commodities in-trend (0.2 each), bonds out → cash. Shareholder yield = dividend yield only (no buyback column exists anywhere in fundamentals; `shareholder_yield.parquet` = ticker × date × div_yield_ttm, 287 payers median 1.45%).

Also fixed on the way: `build_bogle_funds.py` `load_prices` was `shutil.copy2` on the hive dir (broken since migration) → now hive-reads directly; mcap gate coverage measured within the `daily_mcap.parquet` span (2016+) instead of diluting against 1990+ price history (gate passed 0 names before). TMI rebuilt from the hive — the old file was drag-only (1000→997, ret_gross all 0) which had been silently weakening every "vs TMI" comparison in items 8/9.

**Checklist:**
- [x] Asset-class mapping maintained in `gtaa_trend.py` `ASSET_CLASSES` (ETFs verified in hive; grid 1993→2026)
- [x] EW class indices from real ETF closes (no `sector_prices`)
- [x] 10-month SMA trend per class, monthly rebalance, weights lagged (no lookahead)
- [x] GTAA backtest vs TMI same-window (TMI rebuilt real, see above)
- [x] Shareholder yield = dividend yield only (no buyback column; fallback path per plan)
- [ ] Coverage ≥ 500 names — **FAIL: 287 payers** (universe is 365 coverage names; most non-payers are growth/tech)
- [x] `gtaa_sleeve.parquet`, `gtaa_backtest.parquet`, `shareholder_yield.parquet` written

**Do not:** Trend `sector_prices` levels. Do not put GTAA state in the LLM brief.

**Bar:** GTAA vs TMI: maxDD ratio **< 0.70** (2020 and 2022) and full-sample net CAGR within **2 pp** of TMI. Else it is a DD tool, not a return engine.

**Output:** `gtaa_sleeve.parquet`, `shareholder_yield.parquet`.

### 11. Cochrane — Discount Rate Decomposition

**Status:** **Measured (2026-09-01) — coverage bar PASS.** Latest per-ticker implied-r: **2,336 names** with r > 0 (bar ≥ 2,000). CF>DR **0.7%** of rows — discount-rate spread is not a discriminator.

**Checklist:**
- [x] Keep `r = 2·ROE/(P/B+1)` gated ROE>0, P/B>0, r>0 (`4e0d1ca` commit)
- [x] Expand screen: compute implied-r for every name with valid ROE, P/B, price on each filing date — `implied_r_screen.parquet` now **3,925 rows, 54 as-of dates**
- [x] Verify latest implied-r notna count ≥ 2,000 names with r>0 — **2,336 (PASS)**
- [x] If < 2,000: diagnose binding hole… — not needed (bar met)
- [x] CF vs DR: compute `cf_yield = ROE / P/B`, `discount_rate = 2·ROE/(P/B+1)`; report top/bottom tercile in consumers — `implied_r_decomp.parquet` (496 names)
- [x] Do not dump 0–1 scores; word as "CF>DR" / "CF<DR" / "inconclusive" in briefs
- [x] Write updated `implied_r_screen.parquet` / `implied_r_decomp.parquet` (same schema, longer coverage)

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

**Status:** **Measured (2026-09-02) — CPPI maxDD bar PASS, terminal-wealth bar FAIL at every m → CPPI is a floor tool; ERC stays the risk-parity claim, Vince LS stays the leverage claim.** `cppi_backtest.py`: 400 SHARED block-bootstrap paths (21d blocks, seed 0) over rebuilt-real TMI/BPI daily returns (2016-08→2026-08). Books: CPPI floor 0.9×peak with m ∈ {2,3,4} (TMI leg, cash residual), ERC inverse-vol (w_tmi 0.47), HRP (analytic 2-asset reduction = inverse-variance, w_tmi 0.45 — `hrp_weights_from_cov` crashes on 2-asset frames), Vince LS f=1.50 TMI.

| book | median terminal | p05 | median maxDD | median underwater (days) |
|---|---|---|---|---|
| cppi_m2 | 1.23 | 1.08 | 4.3% | 278 |
| cppi_m3 | 1.33 | 1.09 | 6.1% | 303 |
| cppi_m4 | 1.41 | 1.09 | 7.6% | 324 |
| erc | 1.86 | 1.03 | 21.5% | 327 |
| hrp | 1.82 | 1.00 | 21.6% | 338 |
| vincent_ls | 4.37 | 1.73 | 30.1% | 261 |

CPPI keeps DD ≤ 7.6% vs ERC 21.5% — best floor per unit risk — but median terminal 1.23–1.41 misses the 0.8×ERC (1.49) bar even at m=4. Bar result: **FAIL** (both conditions required). Also re-ran `kelly.py ls-vs-erc` on rebuilt indexes: LS median 4.37 vs ERC 1.92 — the earlier 0.995-vs-0.997 result was drag-only shell data and is superseded.

**Checklist:**
- [x] CPPI on TMI: floor = 90% peak NAV, m ∈ {2,3,4}
- [x] 400 block-bootstrap paths, same block size (21d) as the Vince/ERC test; paths SHARED across books
- [x] CPPI vs ERC vs HRP vs Vince LS on median terminal, maxDD, underwater run
- [ ] Multiple start dates — not run (bootstrap paths already resample the full window; single-window caveat documented)
- [x] Cost assumptions: none charged (index-level bootstrap, monthly-equivalent drift; same convention as ls_vs_erc)

**Do not:** Size live books at Vince f=1.50. Do not declare CPPI a winner on one crisis.

**Bar:** CPPI maxDD **< ERC maxDD** and terminal wealth **not** worse than ERC median by 20%. Else ERC stays the risk-parity claim; LS stays the leverage claim. → **Bar FAIL: ERC + LS stand.**

**Output:** `cppi_paths.parquet`.

#### 14b. Spec — CPPI floor overlay on the personal book (narrow)

**Status:** **Spec'd (2026-09-04)** — not implemented.

**Goal:** cap the 9-name universal book's drawdown with the item-14 floor machinery
without giving up the gated-UP terminal wealth. The book is not TMI/BPI: it is a
$300-ish, 9-name, all-equity position whose NAV path is `universal_book_weights`
daily gated wealth (2,799-day common window, 2.85× gated).

**Mechanics (Kouwenberg–Zhu style ratchet, daily, no lookahead, no leverage):**
- NAV_t = book value at close t (holdings × last_close, cash = 0 residual)
- Floor F_t = 0.90 × running peak NAV, ratcheted **up only** (never down)
- Cushion C_t = NAV_t − F_t; equity exposure w_e(t) = min(1, m × C_t / NAV_t),
  m ∈ {2, 3, 4}; the residual (1 − w_e) sits in cash
- **The multiplier m scales the cushion, it is NOT leverage on the book** —
  w_e ≤ 1 always. m=4 is the most aggressive de-risk profile, not borrowing.
- De-risk when NAV threatens the floor (w_e falls), re-risk only when the new
  peak ratchets the floor higher. No intraday; close-to-close only.

**Bar (same convention as 14):** book+CPPI median maxDD **< 0.5 ×** book-alone
maxDD AND median terminal **≥ 0.8 ×** book-alone terminal. Book-alone maxDD is
~20% on the universal path; floor target ≤ ~10%.

**Research leg:** re-use `cppi_backtest.py`'s 400 SHARED block-bootstrap paths —
apply the floor to the gated-UP book wealth path (not raw TMI/BPI) so the floor
is tested against the actual book engine. Single m pass → book tool; both bars
fail → one-line fail, keep the book un-floored.

**Checklist:**
- [ ] `UniversalEngine`/`PortfolioResult` gains `cppi_floor(m, floor=0.9)` method
- [ ] Book run: 400 shared paths, floor overlay, maxDD + terminal per m
- [ ] Narrow: live DAG job `universal_book_cppi` (after `universal_book`) writing
      `universal_book_cppi_weights.parquet` (date × ticker × w_e × cppi_weight)
- [ ] Dashboard tile: book NAV vs floored NAV + current w_e

**Do not:** Call the reactivation "market timing". Do not let the floor gate the
universal rebalance (the §10 gate stays separate). Do not m > 4. Do not apply the
floor to research paths and the book with different cost conventions.

**Output:** `universal_book_cppi_weights.parquet`.

### 15. Merton — ICAPM + Multi-Hedge (and the BL blend)

**Status:** **Spec'd (2026-09-04) — not implemented.** `black_litterman.py` exists. Macro panels are stale or junk (ERP already inside WACC; `exogenous_panel` ~1 month behind prices).

#### 15a. Spec — ICAPM hedge sleeves (broad research leg)

**Goal:** price the personal book's exposure to the three non-market state
variables Merton's ICAPM says a long-horizon investor hedges, and emit sleeves
whose weights are estimated, not asserted.

| sleeve | instrument proxy (from the hive, no new feeds) | why it is the proxy |
|---|---|---|
| bond/TIPS | TIP ETF daily total return | unexpected-inflation + real-rate state var that erodes nominal book |
| currency | UUP (dollar) daily total return | USD shocks hit ADRs in the book (BAYRY, HMC) harder than pure domestics |
| labor-income | SPY sector-neutral construction per Santos–Veronesi 2006 (long high-labour-share sectors vs short low-labour-share, within SPY) | human-capital payoff covariance the ICAPM says matters at horizon |

**Mechanics:**
- Daily returns of the three sleeves from `sector_prices.parquet` + hive ETFs.
- Rolling 252d multivariate regression of book excess returns on
  [market, sleeve1, sleeve2, sleeve3] → hedge coefficients β per sleeve.
- Hedge sleeve weight = −β scaled by inverse-vol, capped: |w_sleeve| ≤ 0.30
  (small book, no naked shorts beyond the pair structure; the sleeve stays
  within the SPY pair, so it is a spread, not a directional bet).
- Weekly rebalance (monthly was the TMI-leg convention); cost 0.10% charged
  on the sleeve turn.
- Backtest on full book window (2,799d): ICAPM-hedged book vs book alone vs
  book + CPPI. **Bar:** hedged median maxDD < book-alone maxDD AND hedged
  terminal ≥ 0.8 × book-alone terminal (same convention as 14). Hedge that
  costs wealth is a failed hedge — go unhedged.

**Checklist:**
- [ ] 3 sleeve return builders (hive-only: TIP, UUP, SPY sector pair)
- [ ] Rolling 252d OLS coefficients, inverse-vol scaling, ±0.30 cap
- [ ] 400 shared bootstrap paths, weekly rebalance, 10bps cost
- [ ] Compare hedged vs unhedged vs CPPI-overlaid on maxDD + terminal
- [ ] Write `icapm_hedge_sleeves.parquet` (date × sleeve × weight) only if the bar passes

**Do not:** Put inflation/oil/CPI in `forecast_llm.py`. Do not invent a
labor-income factor from GICS (Santos–Veronesi construction only). Do not run
the sleeve as the whole book (it is a hedge, max ±30% notional).

#### 15b. Spec — Black-Litterman blend for the personal book (narrow)

**Goal:** fold the *passed* active signals into the universal baseline as
view-tilted weights — the one mechanism that gives the active layers a vote in
**size**, not just a veto gate.

**Inputs:**
- Prior = the gated universal book weights (item 23) — the regret-optimal
  no-lookahead baseline, NOT market-cap (the repo has no clean broad cap-weight
  panel for a 9-name book; UP is the honest prior).
- Views: **only** from Phase-1/2 signals that passed their own bars —
  implied-r (>0 gate, item 11 PASS), QV∩NM quality (item 9), ER hit-edge
  (+6.3pp ✓); dyn weights / meta-labeling / aggregator 0–1 scores stay OUT by
  the existing do-nots.
- View conviction τ: from the signal's own OOS IC (preferred 0.048, peer 0.177,
  pair 0.127, cross 0.025, earnings 0.022 — the measured live ICs, item 3).

**Mechanics (standard BL, one-line closed form):**
- Baseline covariance from 252d book-name returns (hive, Polygon-priority).
- View vector P·q with diagonal uncertainty Ω = diag(τ·P·Σ·Pᵀ) — conviction
  scales the view, measured IC sets τ; no view on a name → posterior = prior.
- Posterior w_BL = MAP of [universal prior, signal views]. Apply the Cover §10
  cost gate to w_BL vs current holdings (same gate as the plain book).
- Emit `bl_book_weights.parquet` (date × ticker × w_BL) + the CVaR/turnover
  comparison vs the ungated-UP book.
- **Bar:** BL book terminal ≥ UP-book terminal on the 400 shared paths AND
  turnover ≤ 2× plain book. A blend that adds turnover without wealth loses.

**Do not:** Feed signals that failed their bars into views. Do not use
dynamic/aggregator scores. Do not let |w_BL − w_UP| exceed 0.05 per name per
month (the tilt must be a tilt, not a rewrite). Do not treat BL as alpha —
it is the mechanism that lets measured signals *adjust* the baseline.

**Checklist:**
- [ ] Prior/covariance/views plumbing on the 9-name book
- [ ] 400 shared paths: UP vs BL (terminal, maxDD, turnover)
- [ ] Live DAG job `universal_book_bl` after `universal_book` (writes
      `bl_book_weights.parquet`, gate-honoring)
- [ ] Dashboard: BL target row next to the UP/current rows

**Outputs:** `icapm_hedge_sleeves.parquet`, `bl_book_weights.parquet`.

**Ordering (per sequencing rules):** 14b (CPPI floor) first — it rides the
existing engine and needs no new data. 15a then 15b order is flexible; 15b can
proceed independently since its sleeve inputs are the same hive data. Neither
touches `forecast_llm.py` or the hive writers.

**Bar:** ICAPM-hedged TMI vs TMI: maxDD ratio **< 0.85** with net CAGR loss **< 1.5 pp/yr**. Else BL stays a view engine, not an ICAPM claim.

**Output:** `icapm_hedge_sleeves.parquet`.

### 23. Cover — Universal Portfolios (pulled forward 2026-09-04)

**Status:** **Measured (2026-09-04)** — research bar PASS; side-info measured (wash); book tool live as weights + sizing plan.

**Core papers:** Cover 1991 *Universal Portfolios* (Math. Finance 1(1):1–29); Cover & Ordentlich 1996 side info (IEEE-IT 42(2):348–363); Ordentlich & Cover 1998 minimax (Math. OR 23(4):960–982) — see `docs/cover_universal_portfolio.md`.

**Engine (`universal_portfolio.py`, refactored to a reusable library):** `UniversalEngine` (exact m=2 via Cover eq. (128) Q-telescope; MC simplex for m≥2, Kalai–Vempala style; telescope identity Ŝₙ = ΣQ self-checks every run; validated exact vs 100k MC within 0.05%), `PortfolioResult` (weights/wealth/gated/trades/stats), `RegimeStates` (tape-level HMM states as side info, reusing `hmm_regime_detection.build_features`), `SizingPlan` (target-vs-holdings trade plan). Zero assumptions, zero lookahead, no shorts (simplex constraint). CLI is a thin wrapper; everything importable.

**Research (shared 400-path, 21d-block bootstrap, seed 0 — same paths as item 14, 2021-08→2026-08):**

| book | median terminal | p05 terminal | median maxDD | median underwater days |
|---|---|---|---|---|
| universal | **1.96** | 1.12 | **20.9%** | 309 |
| universal_side (HMM) | **1.94** | 1.10 | 21.1% | 312 |
| erc | 1.86 | 1.03 | 21.5% | 327 |
| hrp | 1.82 | 1.00 | 21.6% | 338 |
| cppi_m3 | 1.33 | 1.09 | 6.1% | 303 |
| vincent_ls | 4.37 | 1.73 | 30.1% | 261 |

**Bar (same as item 14): median terminal ≥ 0.8×ERC AND median maxDD < ERC → PASS (both universal and universal_side).** Universal beats ERC on both axes (1.96× vs 1.86×; 20.9% vs 21.5%) with no lookahead. **Side-info result: a wash (1.94 vs 1.96, 0.99×)** — on the TMI/BPI index tape the best state-conditional CRP ≈ the best CRP, so HMM state conditioning adds no terminal wealth; the extra regret dimension (d=k(m−1)=3) slightly costs. The side-info claim is honest-but-flat HERE; the states may matter more on the 9-name book. Vince LS still owns the leverage claim (4.37×, 30% DD). Note: 1.96 supersedes the earlier 1.97 figure — the library refactor reports stats off the exact wealth series.

**Book (narrow, personal portfolio, common trading window 2,799 days ≈ 11y, 9 names):** universal (cost-gated, 0.10%) **2.85×** vs equal-weight **2.58×** vs best hindsight name SMCI 13.2×. The honest read: modest +8–11% over the naive no-lookahead baseline; it cannot know SMCI was the winner. Cover §10 rebalance gate: only 24 of 2,799 days trigger (log-wealth gain > log(1+cost)), and the gate *improves* wealth (2.85 vs 2.78 ungated) — the gate is not just cost control, it filters rebalance noise. **Sizing wired:** `SizingPlan` compares latest gated weights vs `portfolio_holdings.parquet` (percent-form normalized) → per-name delta shares at last close → `universal_sizing_plan.parquet`; latest plan trims BAYRY/CAG/HMC/KHC/T, adds HPQ/PFE/SMCI (SMCI +0.62 shares). Latest weights → `universal_book_weights.parquet`; dashboard-exposed.

**Checklist:**
- [x] Exact Dirichlet(1/2) m=2 engine + telescope self-check; MC engine validated vs exact
- [x] Research: 400 shared bootstrap paths, universal vs erc/hrp/cppi_m3/vincent_ls → `universal_paths.parquet`
- [x] Book: daily universal weights for BAYRY,CAG,HMC,HPQ,KHC,MOS,PFE,SMCI,T → `universal_book_weights.parquet`
- [x] Cover §10 cost gate (rebalance iff ΔlnW > ln(1+c)) — measured, improves results
- [x] Side-information variant (Cover–Ordentlich; HMM states = `RegimeStates`, fitted on the ORIGINAL tape, resampled with the same block draws — no lookahead) — **measured: 0.99× plain, flat on the index tape**
- [x] Sizing plan vs holdings (percent normalized, delta shares, gate-honoring) → `universal_sizing_plan.parquet`, wired into the `universal_book` DAG job

**Do not:** Claim universal > oracle at finite n (that's a theorem violation). Do not trade the book at Ungated daily weights (gate exists). Do not treat this as alpha — it is the regret-optimal no-lookahead baseline to beat. Do not claim HMM side-info adds value on the index tape (measured flat).

**Outputs:** `universal_paths.parquet`, `universal_book_weights.parquet`, `universal_sizing_plan.parquet`.

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

### 24. Macro — UMKC/MMT sectoral balances + KC Fed labor/natural rate (pulled forward 2026-09-05)

**Status:** **Spec'd (2026-09-05)** — not implemented. Live layer is Keen/Minsky demand (`macro_fragility.py`) + oil/CPI/FFR supply (`macro_shock.py`) + a calendar that is mostly option-expiry (`economic_calendar.py`). Sector/macro stay **out** of `forecast_llm.py` (measured 2026-08-30).

**Live state (do not invent):**
- `macro_fragility.parquet` last **2026-01-01** (two quarters stale vs prices): debt/GDP **3.63**, impulse **0.209**, p_stress **0**, minsky_pctile **0.75**, `danger_zone=crisis_band`. Calm + levering is the FIH reading — the label is the bug (see below).
- `macro_shock.parquet` last **2026-08-01**: oil 12m **+29%**, inflation surprise ~0, real rate ~0, `shock_zone=elevated`.
- FRED cache: TCMDO, GDP, M2V, CPI, FEDFUNDS, WTI — **no** private-vs-federal split, **no** sectoral balances, **no** KC Fed LMCI / natural rate.

**The load-bearing error (Keen × Wray):** `TCMDO` is **all sectors** (households + business + **federal**). Keen's FIH variable is **private** debt; Wray's "Kansas City" MMT says a floating-rate sovereign **cannot** be insolvent in its own currency — federal debt/GDP is not a Minsky crisis stock. Mixing them is why `danger_zone=crisis_band` can fire on Treasury issuance. Split TCMDO (or CMDEBT / household+nonfin business) vs FYGFD (federal). Private impulse stays the fragility claim; federal net injection is the **sectoral-balance** claim.

**What to add (FRED + KC Fed public series, no new paid feeds):**

1. **Sectoral balances (Godley / UMKC-Wray).** Private surplus ≈ public deficit + current account. Series: federal receipts/expenditures or net lending (NCBEILQ027S / FGEXPND / FGRECPT), current account. When the *private* sector is in deficit, that **is** the Keen impulse in MMT clothing. Output: `sectoral_balances.parquet` (date, private_balance, public_balance, cad).
2. **Private-debt impulse (Keen, corrected).** Recompute `debt_impulse` on private debt only. Bar: private impulse vs NBER recession starts must not be worse than the all-sector series; if the Jan-2026 `crisis_band` **flips** after the split, that is a finding, not a patch — print both.
3. **Inflation constraint, not solvency.** MMT: the binder is real resources + inflation, not the bond vigilante. Keep `inflation_surprise`; add core PCE and the **KC Fed Model-Based Natural Unemployment Rate** + unemployment gap (Glover/Oliyide Charting the Economy, Aug 2026: Taylor-rule FFR "should probably be higher — by how much is uncertain"). Real rate in `macro_shock` is FFR−CPI, not r*. Do not treat the sign of FFR−CPI as a natural-rate claim.
4. **Labor buffer (JG proxy, not a JG).** Do not implement a job guarantee as a portfolio weight. Do ingest the **KC Fed Labor Market Conditions Index** (24-series LMCI — Schmid 2026 speech: cooling, still a touch above average). Bundick (KC Fed) / Cairó / Petrosky-Nadeau FEDS 2025-068: shortfalls vs deviations on maximum employment. Slack = inflation buffer in the UMKC buffer-stock story.
5. **Tenth District (optional, local).** KC Fed manufacturing survey already sits in the Board MPR regional average. Not a book signal until it has a bar vs TMI.

**Bar:** (a) private-debt series exists and is DATE-native quarterly; (b) `danger_zone` is computed from **private** impulse, not TCMDO-all; (c) sectoral balances last date within 1 quarter of FRED; (d) **no** new field in the LLM brief. If private impulse fails to lead recessions vs the current series, keep TCMDO as a published alternative column — do not silently swap.

**Do not:** Put CPI/oil/MMT slogans in `forecast_llm.py`. Do not treat US federal debt/GDP as a solvency veto on T / RF / KEY. Do not size the book on the Minsky pctile (it's a regime flag, not a weight). Do not claim the job guarantee is in the engine.

**Outputs:** `sectoral_balances.parquet`, `macro_fragility.parquet` (private + all-sector columns), optional `kc_labor.parquet` (LMCI + U-gap).

**Ordering:** after item 15 sleeves (TIP already is the inflation hedge); does not block Cover/sizing. UMKC/Wray is the *interpretation* of the fiscal columns; Keen stays the *private-debt* mechanic.

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