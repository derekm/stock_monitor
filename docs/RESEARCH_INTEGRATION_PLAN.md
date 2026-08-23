# Research Integration Plan: 22 Scholars → StockMonitor Stack

**Goal:** Systematically integrate each researcher's core contributions into our codebase, with measurable upgrades to specific files. Priority deep dives (1–7) first; remaining 15 in maintenance/extension cycles.

---

## Phase 1: Priority Deep Dives (Weeks 1–14)

### 1. Fama/French + Novy-Marx — Factor Construction Validation
**Target files:** `peer_analytics.py`, `preferred_metrics.py`, `factor_rotation_defense.py`, `signal_aggregator.py`
**Core papers:** FF 1993/2015 (5-factor), Novy-Marx 2013 (gross profitability), 2014 (quality)
**Deliverables:**
- [x] Replicate FF 5-factor + momentum on our `daily_prices` universe → `ff5_factors.parquet` (daily)
- [x] Validate our quality gate (ROE>12%, ROIC>10%, D/E<1.5, trifecta≥2) against Novy-Marx "quality" (gross profit/assets + low accruals + safe leverage) → `docs/QUALITY_GATE_COMPARISON.md`
- [x] Map each signal family (preferred/peer/cross/pairs/earnings) to FF factor loadings → `signal_factor_loadings.parquet`
- [x] Update `signal_aggregator.py` OOS-IC weights with factor-adjusted IC (residualize signals on FF5+MOM) — added `--use-residuals` flag
- [ ] Add `factor_attribution.py` script: daily factor decomposition of portfolio/aggregate returns

**Success metric:** Signal IC improves ≥0.02 after factor adjustment; quality gate overlap with Novy-Marx ≥80%

---

### 2. Ilmanen + Ang — Expected Return Framework
**Target files:** `implied_r_screen.py`, `damodaran_quality.py`, `preferred_metrics.py`, `macro_fragility.py`
**Core papers:** Ilmanen *Expected Returns* (2011), Ang *Asset Management* (2014), Ilmanen et al. "Carry" (2013)
**Deliverables:**
- [x] **ACTIVE** Implement Ilmanen's 4-pillar expected return decomposition for every ticker:
  - Carry (yield + roll-down)
  - Value (mean reversion)
  - Momentum (trend)
  - Defensive (low risk anomaly)
  → `expected_returns_decomp.parquet` (ticker × date × pillar)
- [ ] Replace `implied_r_screen.py` single-number ICC with Ilmanen's **cash-flow vs. discount-rate** decomposition (Cochrane 2011) → `implied_r_decomp.parquet`
- [ ] Integrate Ang's **factor timing** framework: regime-conditional factor premia → `regime_factor_premia.parquet`
- [ ] Add carry metrics to `macro_fragility.py` (sovereign carry, credit carry, equity carry)
- [ ] Build `expected_return_report.py`: dashboard tile showing forward-looking ER by pillar/sector

**Success metric:** Out-of-sample direction accuracy of pillar-weighted forecast > equal-weight benchmark by ≥5%

---

### 3. Asness/Pedersen — Signal Aggregation + Cost-Aware Weighting
**Target files:** `signal_aggregator.py`, `signal_model.py`, `cost_model.py`, `shadow_book.py`
**Core papers:** Asness et al. "Value and Momentum Everywhere" (2013), Pedersen "Efficiently Inefficient" (2015), AQR "Factor Timing" (2020)
**Deliverables:**
- [ ] Replace static OOS-IC weights in `signal_aggregator.py` with **Pedersen's dynamic weighting**:
  - Weight ∝ IC / (turnover × cost) × regime-confidence
  - Add horizon-aware decay (signals have different half-lives)
  → `signal_weights_dynamic.parquet` (daily)
- [ ] Implement **cost-aware optimization** (Pedersen Ch. 9): maximize ∑ wᵢ·ICᵢ - λ·wᵀΣw - τ·turnover(w)
  - Use our `cost_model.py` (10bps + borrow) for τ
  - Solve via quadratic programming → `optimal_signal_weights.parquet`
- [ ] Add **signal decay curves** per family (preferred: slow; earnings: fast; technical: fastest) → `signal_decay_params.json`
- [ ] Integrate with `shadow_book.py`: simulate paper portfolio using dynamic weights + cost model → `shadow_dynamic.parquet`

**Success metric:** Dynamic-weighted portfolio Sharpe > static OOS-IC portfolio by ≥0.15 after costs

---

### 4. Taleb/Spitznagel/Haghani — Hardened Taleb Layer
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
**Target files:** `subindustry_regime.py`, `peer_analytics.py`, `cross_section.py`, `signal_model.py`, `hmm_regime_detection.py`
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
**Target files:** `rebalance_calendar.py`, `vol_target.py`, `kelly.py`, `portfolio_optimization.py`, `risk_parity_analytics.py`
**Core papers:** Hoffstein "Rebalancing Luck" (2019), "Sequence Risk" (2020), Vince *Leverage Space Trading Model* (2009), *The Leverage Space Model* (2013)
**Deliverables:**
- [ ] **Rebalancing luck quantification**: Monte Carlo `rebalance_calendar.py` over all possible rebalance days in quarter → `rebalance_luck_distribution.parquet`
- [ ] **Multi-day glide optimization**: Extend 5-day linear glide to **Hoffstein's optimal glide path** (minimize tracking error variance) → `optimal_glide_schedule.parquet`
- [ ] **Sequence risk metrics**: Add to `perf_metrics.py` — **sequence risk score** (correlation of returns with withdrawal phase), **conditional drawdown at risk** (CDaR)
- [ ] **Leverage Space sizing**: Implement Vince's multi-asset optimal f (joint distribution of returns, not marginal Kelly) → `leverage_space_allocation.parquet`
- [ ] **Path-dependent Kelly**: Replace `kelly.py` single-period with multi-period Kelly (accounting for volatility drag) → `multi_period_kelly.parquet`

**Success metric:** Glide path reduces rebalancing luck std by ≥40%; Leverage Space allocation dominates ERC risk parity in Monte Carlo

---

### 7. Lo/Amodei — Adaptive Markets + LLM Forecasting
**Target files:** `hmm_regime_detection.py`, `statistical_profiler.py`, `forecast_granite.py`, `granite_daily.py`, `regime_calibrate.py`, `regime_serving.py`
**Core papers:** Lo *Adaptive Markets Hypothesis* (2004/2017), Amodei et al. *Constitutional AI* (2022), Granite TTM papers (IBM 2023-2024)
**Deliverables:**
- [ ] **Adaptive HMM**: Extend `hmm_regime_detection.py` with Lo's **time-varying transition probabilities** (regime persistence changes with volatility) → `adaptive_hmm_states.parquet`
- [ ] **Population dynamics**: Add `statistical_profiler.py` metric: **regime population fitness** (fraction of tickers in each regime, evolution over time) → `regime_population.parquet`
- [ ] **LLM forecasting integration**: Prototype `forecast_granite.py` → `forecast_llm.py` using fine-tuned LLM (e.g., FinGPT, BloombergGPT, or local Llama) for directional + narrative forecasts
- [ ] **Uncertainty calibration**: Upgrade `regime_calibrate.py` with **conformal prediction** (distribution-free prediction intervals) + Amodei's **constitutional uncertainty** (model expresses "I don't know") → `conformal_bands.parquet`
- [ ] **Regime-selected ensemble**: Enhance `regime_serving.py` with **dynamic model weighting** (Lo's evolutionary weight update based on recent regime performance) → `ensemble_weights.parquet`

**Success metric:** Adaptive HMM regime persistence correlation with realized vol > 0.6; conformal bands achieve 90% coverage

---

## Phase 2: Core Extensions (Weeks 15–26)

### 8. Jegadeesh/Titman — Momentum Foundations
**Target:** `momentum_analytics.py`, `fractal_windows.py`, `backtest_price_vs_momentum.py`
**Deliverable:** Replicate original 12-2 momentum; compare vs our fractal stack; document where fractal adds value

### 9. Gray/Vogel — Quantitative Value/Momentum
**Target:** `preferred_metrics.py`, `inclusion_criteria.py`, `dual_screen_analysis.py`
**Deliverable:** Implement Alpha Architect's exact screens (QV: EV/EBITDA + GP/A + low leverage; QM: 12-1 momentum + quality); A/B test vs our dual-pass

### 10. Faber — GTAA + Shareholder Yield
**Target:** `arista.py`, `macro_shock.py`, `cross_asset_analysis.py`, `build_bogle_funds.py`
**Deliverable:** Add GTAA sleeve (trend-following across 10 asset classes via sector ETFs); shareholder yield screen

### 11. Cochrane — Discount Rate Decomposition
**Target:** `implied_r_screen.py`, `macro_fragility.py`, `damodaran_quality.py`
**Deliverable:** Full CF/DR decomposition for every ticker; link to macro shock indicators

### 12. Baker/Wurgler — Sentiment + Catering
**Target:** `filings_sentiment.py`, `estimate_revisions.py`, `hmm_regime_detection.py`
**Deliverable:** Sentiment index from 8-K + earnings calls; regime-sentiment interaction

### 13. Moskowitz — Time-Series Momentum
**Target:** `fractal_windows.py`, `ride_longevity.py`, `momentum_analytics.py`
**Deliverable:** TSMOM factor (12-month lookback, vol-scaled); compare vs cross-sectional momentum

### 14. Perold/Sharpe — CPPI + Risk Parity
**Target:** `vol_target.py`, `risk_parity_analytics.py`, `portfolio_optimization.py`
**Deliverable:** CPPI floor + multiplier optimization; compare ERC vs HRP vs CPPI

### 15. Merton — ICAPM + Multi-Hedge
**Target:** `black_litterman.py`, `portfolio_optimization.py`, `regime_aware_constraints.py`
**Deliverable:** ICAPM hedge portfolios (inflation, labor income, currency); integrate with BL views

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
| **Gate 1 (Week 4)** | FF5 replication validated; quality gate comparison done | **✅ PASSED** — Continue Phase 1 |
| **Gate 2 (Week 8)** | Dynamic signal weighting beats static; expected return decomposition live | Continue Phase 1 |
| **Gate 3 (Week 14)** | Taleb layer hardened; barbell portfolio backtested | Enter Phase 2 |
| **Gate 4 (Week 26)** | ML regime upgraded; sequence risk metrics live | Enter Phase 3 |
| **Gate 5 (Week 38)** | LLM forecaster production; Leverage Space complete | Maintenance mode |

---

## Quick Start (This Week)

```bash
# 1. FF5 replication on our universe
python -c "
from src.analytics.factor_library import compute_ff5
ff5 = compute_ff5('daily_prices.parquet')
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

*Plan version: 2026-08-22. Update after each gate review.*