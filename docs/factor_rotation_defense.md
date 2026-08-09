# factor_rotation_defense.py

Defensive factor rotation — rotate into the currently defensive factor
sleeve (quality/value/low_vol/dividend) based on regime + momentum signals,
and defend against regime transitions.

## Why it exists (rationale)

The macro layer (`macro_fragility.py`, `macro_shock.py`) gives a high-level
regime view. The factor layer translates that into **sleeve-level** allocation:
rotate the defensive allocation into the sleeve that is currently both
defensive *and* showing positive momentum, while shrinking exposure when
the regime is `high_vol_stress`.

## Formulas

**Factor sleeve returns (equal-weight, monthly):**

For each factor group $g$ (from `factor_group_members.csv`), monthly
equal-weight return:

$$
r_{g,t} = \frac{1}{|G_t|} \sum_{i \in G_t} \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)
$$

**Rotation score (per sleeve, monthly):**

$$
\text{score}_g(t) = \text{mom}_{12,g}(t) \cdot \mathbb{1}[\text{regime} \neq \text{high\_vol\_stress}]
$$

Only rotate into a sleeve if it has positive 12m momentum AND the current
regime is NOT `high_vol_stress` (regime from `hmm_regime_states.csv`).

**Defensive allocation weight:**

$$
w_g(t) = \frac{\text{score}_g(t)_+}{\sum_h \text{score}_h(t)_+} \times (1 - \alpha \cdot p\_stress)
$$

where $p\_stress$ = HMM stress posterior (from `hmm_regime_states.csv`);
$\alpha = 0.5$ scales the allocation down as stress rises.

## Outputs

- `factor_rotation_weights.csv` — per sleeve per month: `date, sleeve, weight, score, mom12, regime, p_stress`
- `factor_rotation_performance.csv` — backtest: `sleeve, total_return, max_dd, sharpe, n_months`
- `factor_sleeve_returns.csv` — monthly return series per sleeve

## Usage

```bash
python factor_rotation_defense.py --save
```

Wired into `run_daily_automation.py` as `taleb_factor_rot`; feeds export.

(Schema families: weights_performance / base_table — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [macro_fragility.md](macro_fragility.md) — p_stress input
- [hmm_regime_detection.md](hmm_regime_detection.md) — regime input
- [factor_panel.md](factor_panel.md) — sleeve construction
- [portfolio_optimization.md](portfolio_optimization.md) — weight consumer
- [vol_target.md](vol_target.md) — vol target consumer