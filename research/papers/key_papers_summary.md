# Key Papers from Quant Research Conversation

## Core Researchers & Labs to Follow

### Foundational Research Hygiene (Non-negotiable Best Practice)
- **López de Prado** — *Advances in Financial Machine Learning*; "The 10 Reasons Most Machine Learning Funds Fail"
  - Meta-labeling, combinatorial purged CV, deflated Sharpe, fractional differentiation, HRP
- **Harvey, Liu, Zhu** — "...and the Cross-Section of Expected Returns" and follow-ons on multiple testing
- **Bailey, Borwein, López de Prado, Zhu** — Pseudomathematics / backtest overfitting
- **Arnott, Harvey, Markowitz-related** — Critiques of naive factor/ML research

### Modern Empirical Asset Pricing + ML
- **Gu, Kelly, Xiu** — "Empirical Asset Pricing via Machine Learning" and autoencoder / conditional beta follow-ups
- **Kelly, Pruitt, Su** — Instrumented PCA (IPCA)
- **Kozak, Nagel, Santosh** — Shrinkage of the stochastic discount factor / "Interpreting Factor Models"
- Recent work on deep factors, attention/transformers for returns, foundation-model style pretraining (ICAIF, arXiv q-fin + cs.LG 2022-2025)

### Microstructure, Impact, Short-horizon Prediction
- **Cont, Kukanov, Stoikov** — Price impact / order flow
- **Donier, Bonart, Mastromatteo, Bouchaud** — Quadratic impact, latent liquidity
- **Cartea, Jaimungal et al.** — Algorithmic trading / market making books and papers
- **Bacry, Muzy, Filimonov, Sornette** — Hawkes-process market models
- **Huang, Lehalle, Rosenbaum** — Queue-reactive models
- **Easley, López de Prado, O'Hara** — VPIN and order-flow toxicity literature

### Execution, Market Making, RL
- **Almgren-Chriss** and successors; **Obizhaeva-Wang**; **Gatheral**; **Carta-Jaimungal** optimal execution
- **Nevmyvaka, Feng, Kolm, Ritter, Halperin** — RL for execution and MM
- Multi-agent RL and simulated LOBs for policy stress-testing

### Leading Edge / Unevenly Adopted (Future Practice Candidates)
- **Causal ML & Invariant Prediction** in finance (avoiding spurious cross-sectional patterns)
- **Conformal Prediction / Distribution-free Uncertainty** for position sizing and risk
- **Operator / Signature Methods and Rough Path / Rough Volatility** (Lyons, Gatheral, Bayer, Friz)
- **Graph Neural Nets & Relational Inductive Bias** across assets, supply chains, common ownership
- **Foundation Models / Large Time-series Models** adapted to tick/bar data (with leakage controls)
- **Differentiable Portfolio Layers & End-to-end Alpha→Portfolio Training** (risk and cost inside network)
- **Online / Continual Learning with Explicit Concept-drift Detection** and champion-challenger monitors
- **Synthetic Data & Calibrated Market Simulators** (GANs, diffusion, agent-based) for rare-event testing
- **Alternative Data Fusion with Strict PIT and Entity-Resolution Pipelines**
- **Bayesian Deep Learning / Ensembles** for uncertainty-aware alpha and position sizing
- **Market-impact-aware and Game-theoretic Execution** when not pure price taker
- **Robust Optimization and DRO** for portfolio construction under model uncertainty

## Priority Deep Dive Researchers (Phase 1)

### 1. Fama/French + Novy-Marx — Factor Construction Validation
- **Fama & French** — "The Cross-Section of Expected Stock Returns" (1992); "A Five-Factor Asset Pricing Model" (2015)
- **Novy-Marx** — "The Other Side of Value: The Gross Profitability Premium" (2013); "Quality Investing" (2014)
- **Implementation:** `factor_library.py` — FF5+MOM + Novy-Marx quality factors on our universe

### 2. Ilmanen + Ang — Expected Return Framework
- **Ilmanen** — *Expected Returns* (2011); "Carry" (2013); "Expected Returns: An Investor's Guide to Harvesting Market Rewards"
- **Ang** — *Asset Management: A Systematic Approach to Factor Investing* (2014); "Factor Timing" (2020)
- **Implementation:** `expected_returns.py` — 4-pillar decomposition (carry/value/momentum/defensive)

### 3. Asness/Pedersen — Signal Aggregation + Cost-Aware Weighting
- **Asness, Moskowitz, Pedersen** — "Value and Momentum Everywhere" (2013)
- **Pedersen** — *Efficiently Inefficient* (2015); AQR "Factor Timing" (2020)
- **Target:** Dynamic IC-weighted signal aggregation, cost-aware optimization, signal decay curves

### 4. Taleb/Spitznagel/Haghani — Hardened Taleb Layer
- **Taleb** — *Statistical Consequences of Fat Tails* (2020); *The Black Swan*; *Antifragile*
- **Spitznagel** — *Safe Haven: Investing for Financial Storms* (2020); "Tail Hedging"
- **Haghani & White** — *The Missing Billionaires: A Guide to Better Financial Decisions* (2023)
- **Target:** Tail index (Hill), fragility veto, barbell construction, leverage space

### 5. López de Prado — ML Regime Work Upgrade
- **López de Prado** — *Advances in Financial Machine Learning* (2018): CPCV, meta-labeling, regime clustering, triple-barrier
- **Target:** Meta-labeling, CPCV, hierarchical risk parity + regime clustering, triple-barrier labeling

### 6. Hoffstein/Vince — Sequence Risk + Leverage Space
- **Hoffstein** — "Rebalancing Luck" (2019); "Sequence Risk" (2020)
- **Vince** — *Leverage Space Trading Model* (2009); *The Leverage Space Model* (2013)
- **Target:** Rebalancing luck quantification, glide optimization, sequence risk, leverage space sizing

### 7. Lo/Amodei — Adaptive Markets + LLM Forecasting
- **Lo** — *Adaptive Markets Hypothesis* (2004/2017)
- **Amodei et al.** — *Constitutional AI* (2022); Granite TTM papers (IBM 2023-2024)
- **Target:** Adaptive HMM, population dynamics, LLM forecasting, conformal prediction

## Venues to Monitor
- arXiv: q-fin.TR, q-fin.CP, q-fin.ST, q-fin.PM, stat.ML, cs.LG
- SSRN (Harvey, AQR, academic finance)
- ACM ICAIF proceedings
- NeurIPS/ICML/KDD workshops on AI in Finance
- Journal of Financial Data Science, Quantitative Finance, Market Microstructure and Liquidity, Journal of Portfolio Management

## Practical Ingestion Order for Agent Pipeline
1. Overfitting/validation stack (López de Prado, Harvey et al.)
2. Classical + ML asset pricing (Gu-Kelly-Xiu, IPCA, factor literature)
3. Microstructure & impact (Bouchaud, Cont, Cartea-Jaimungal, LOB models)
4. Execution/RL/costs
5. Newer uncertainty, causal, graph, signature, foundation-model papers — always re-checked under validation stack from step 1

## MX550 (2GB VRAM) Implementation Strategy
- CPU-first tabular models (LightGBM/sklearn) for most alpha work
- Small PyTorch nets with float16, tiny batches, explicit cuda.empty_cache()
- Avoid big Transformers, large LOB image models
- Priority: LightGBM → Meta-label MLP → Tiny TCN → Tiny Transformer (d_model=32 max)