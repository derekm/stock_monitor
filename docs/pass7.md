# pass7.py — experiment-design matrix for regime-selected models

Tests whether pass6's per-regime best-config findings are ROBUST across
experiment designs — the "several different experiment designs with different
mixes" idea — and whether regime-aware training scheduling helps.

## Formulas

**Global temporal split (per arm):**

For each arm $a \in \{\text{boundary}, \text{composition}, \text{lr}, \text{freshness}\}$:

$$
\text{boundary}_a = \text{split\_frac}_a \times T
$$

where $T$ = total history length. Each arm uses its own `split_frac` value.

**Per-regime persistence baseline (same as pass6):**

$$
\text{pers\_dir}_r = \frac{1}{N_{\text{test}}} \sum_{i \in \text{test}_r} \mathbb{1}[y_i \cdot y_{i-1} > 0]
$$

computed on the SAME test windows per regime $r$ for each arm.

**OOS direction accuracy per cell (arm × ticker × regime × config):**

$$
\text{dir\_acc}_{a,t,r,c} = \frac{1}{N_{\text{test}}} \sum_{i \in \text{test}} \mathbb{1}[\hat{y}_i \cdot y_i > 0]
$$

**Direction excess over regime persistence baseline:**

$$
\text{excess}_{a,t,r,c} = \text{dir\_acc}_{a,t,r,c} - \text{pers\_dir}_r
$$

**Best config per arm (aggregation across regimes):**

For each arm $a$, find the config $c^* = (\text{steps}, \text{cap})$ that maximizes
mean excess across regimes (weighted by test window count):

$$
c^*_a = \arg\max_c \frac{1}{R} \sum_{r} \text{excess}_{a,r,c}
$$

**Arm summary statistics:**

For each arm $a$, report:
- $n$ = number of cells evaluated
- $\text{mean\_excess}_a = \frac{1}{n} \sum \text{excess}$
- $\text{max\_excess}_a = \max \text{excess}$
- $\text{best\_config}_a = \text{most common best config}$ across regimes

## Arms (all with pass6 honesty rules: shared global boundary per arm, 96d embargo, per-regime persistence baseline, IBM-base-only fine-tuning)

| Arm | Parameter | Values | Question |
|---|---|---|---|
| **boundary** | split_frac | {0.55, 0.70, 0.85} | Does the best config survive a different train/test boundary year? |
| **composition** | composition | {pure, all} | Does regime specialization beat more training data? |
| **lr** | lr | {gd.LR (1e-4), 5e-5} | Is the finding lr-sensitive? |
| **freshness** | window selection | {full, ~10y} | Does "hold off trainings until the trend switches back" help? |

Every unique $(\text{split\_frac}, \text{steps}, \text{cap}, \text{lr}, \text{composition})$ cell is trained ONCE
and tagged with every arm it belongs to (no duplicate training).

## Outputs

- `/tmp/pass7_results.jsonl` — append-only, resumable via `--resume`
- `regime_model_matrix.csv` — every cell result (arm, ticker, regime,
  split_frac, steps, cap, lr, composition, dir_acc, pers_dir, mape)
- `regime_model_matrix_summary.csv` — per-arm mean/max OOS dir excess +
  most-common best config

## Usage

```bash
python pass7.py --tickers AEP,NVR --arms boundary composition lr freshness
python pass7.py --resume --max-experiments 20
python pass7.py --quick
```

(Schema family: Forecast / anomaly — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [pass6.md](pass6.md) — the single-design pass this generalizes (shared machinery)
- [pass5.md](pass5.md) / [regime_forecast.md](regime_forecast.md) — the honest-OOS harness
- [pass8.md](pass8.md) — RPT base pre-training (extends arms with `base` dimension)