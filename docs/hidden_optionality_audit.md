# hidden_optionality_audit.py

Decision-flip audit — the pathology that forced every numeric driver in
`buy_candidates.py` to become noise-robust.

## Why it exists (rationale)

The original `buy_candidates.py` had hard thresholds: momentum > 0.03,
factor > 0.02, etc. A hidden-optionality audit showed that a **hair of noise**
crossing these knife-edges flips inclusion decisions. The audit proved:

- Hard regime cliff: stress label flip → 28.4% of decisions flipped
- Momentum cliff: 6.8% flipped at the 0.03 boundary
- Factor cliff: 5.1% flipped at the 0.02 boundary

The fix: **noise-robust decisions** — every numeric driver's contribution is
now the **noise-convolved expectation** over its estimation error, not a
hard threshold.

## Formulas

**Soft stress posterior (replaces hard label):**

$$
p_stress = P(state = high_vol_stress | F_t)
$$

from the HMM forward-backward algorithm (`hmm_regime_detection.py`).

**Stress haircut (applied to composite score):**

$$
score \leftarrow score - 0.08 \times p_stress
$$

The coefficient 0.08 was chosen so that p=1 (certain stress) gives the same
penalty as the old hard cliff. No cliff — the penalty scales continuously.

**Noise-convolved expectation (per driver):**

For each driver $x$ with estimated value $\hat{x}$ and estimation error
$\sigma_x$ (cross-sectional std / 4):

$$
contribution = \mathbb{E}_{z \sim \mathcal{N}(0, \sigma_x)}[f(\hat{x} + z)]
$$

where $f$ is the piecewise-linear driver function (e.g., momentum step
function). Computed analytically by integrating over the Gaussian error.

**Steps configs (single source of truth for thresholds):**

| Driver | Config | Old threshold | Noise-robust form |
|---|---|---|---|
| Momentum | `MOMENTUM_STEPS` | 0.03 | `_step_expectation(mom, sig, 0.0, 10)` |
| Factor | `FACTOR_STEPS` | 0.02 | `_step_expectation(factor, sig, 0.0, 10)` |
| Composite | `COMPOSITE_STEPS` | 0.0 | `_step_expectation(comp, sig, 0.0, 10)` |
| Residual momentum | `RESID_MOM_STEPS` | 0.0 | `_step_expectation(resid, sig, 0.0, 10)` |
| Liquidity | `LIQUIDITY_STEPS` | 0.0 | `_step_expectation(liq, sig, 0.0, 10)` |
| Skew | `SKEW_STEPS` | 0.0 | `_step_expectation(skew, sig, 0.0, 10)` |

`sig=0` reproduces the old exact thresholds (backward compatible).

**Decision flip measurement (audit protocol):**

For each decision $d$ and perturbation $\epsilon \sim \mathcal{N}(0, \sigma)$
applied to every driver:

$$
flip = \mathbb{1}[decision(\hat{x} + \epsilon) \neq decision(\hat{x})]
$$

Measured: hard regime cliff → 28.4% flips; momentum → 6.8%; factor → 5.1%.
After noise-robust: <1% flips at same noise level.

## Related

- [buy_candidates.md](buy_candidates.md) — the decisions being audited
- [hmm_regime_detection.md](hmm_regime_detection.md) — stress posterior source
- [forecast_granite.md](forecast_granite.md) — Forecasting-Paradox upgrades