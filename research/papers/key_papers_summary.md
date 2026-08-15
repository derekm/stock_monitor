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