# pass6.py — regime-SELECTED Granite-TTM models (one model per regime)

pass5 proved Granite-TTM is a direction forecaster (OOS). `regime_forecast.py`
showed persistence itself is regime-dependent — so a single global model is
suboptimal. pass6 trains **one model per HMM regime**, sweeps training
parameters per regime, and lets the current regime *select* its dedicated
model + config. This replaces regime-*gated* trust adjustments with
regime-*selected* models.

## Protocol (honesty rules)

1. **Regime tagging** — every window is tagged by the HMM regime in force at
   its forecast point (majority of the last 20 trading days before the context
   end; same convention as `regime_forecast.py`).
2. **Global temporal split** — ALL regimes share one boundary
   (`--split-frac`, default 0.7 of history). Train windows' targets end before
   the boundary; test windows' forecast points start only after a **96-day
   embargo** (`GAP_DAYS = HORIZON`). Verified strictly disjoint per regime
   (max train target end < min test forecast point) — no leakage between any
   regime's train and test sections, and no cross-regime leakage either.
3. **Per-regime models from the IBM base only** (`pretrained=False`) — no
   full-history checkpoint contamination.
4. **Regime-specific persistence baseline** — computed on the SAME test
   windows per regime. Direction accuracy must beat THAT baseline, not 50%.
5. **Minimum test size** — regimes with < 30 test windows are skipped (too
   thin to claim).
6. **Test context may overlap the boundary** — that mirrors live use, where
   recent context is always available; only TARGETS must be disjoint.

## Formulas

**Global temporal split:**

```
boundary = split_frac * T  (T = total history length)
train targets end < boundary
test forecast points > boundary + GAP_DAYS (96 days = HORIZON)
```

**Per-regime persistence baseline:**

$$
\text{pers\_dir} = \frac{1}{N_{\text{test}}} \sum_{i \in \text{test}} \mathbb{1}[y_i \cdot y_{i-1} > 0]
$$

**OOS direction accuracy per cell:**

$$
\text{dir\_acc} = \frac{1}{N_{\text{test}}} \sum_{i \in \text{test}} \mathbb{1}[\hat{y}_i \cdot y_i > 0]
$$

**Direction excess over persistence (selection objective):**

$$
\text{excess} = \text{dir\_acc} - \text{pers\_dir}
$$

**Per-regime best config selection:**

$$
\text{best\_config} = \arg\max_{\text{steps, cap, lr}} \text{excess}(\text{steps, cap, lr})
$$

evaluated independently per (ticker, regime).

**Per-regime parameter sweep grid:**

| Parameter | Values |
|---|---|
| steps | 3000, 6000 |
| cap | 100, 200 |
| lr | gd.LR (1e-4), 5e-5 |
| head_only | True, False |
| exog | True, False |

Selection objective: **max OOS direction excess over the regime's persistence
baseline** — the honest "does this model add skill in this regime" measure.

## Outputs

- `/tmp/pass6_results.jsonl` — append-only, resumable via `--resume`
- `regime_model_oos.csv` — every (ticker, regime, steps, cap, lr) result
- `regime_model_best.csv` — best config per (ticker × regime)

## Usage

```bash
python pass6.py --tickers AEP,NVR,FICO --steps 3000 6000 --caps 100 200
python pass6.py --tickers AEP --regimes high_vol_stress
python pass6.py --tickers AEP --resume --max-experiments 20
```

(Schema family: Forecast / anomaly — see [SCHEMAS.md](SCHEMAS.md).)

## Related programs

- [pass5.md](pass5.md) — the base honest-OOS harness (trainlast protocol)
- [regime_forecast.md](regime_forecast.md) — per-regime baseline measurement that motivated this
- [hmm_regime_detection.md](hmm_regime_detection.md) — produces the regime labels consumed here
- [pass5_sweep.md](pass5_sweep.md) — the global param sweep this complements (per-regime)
- [pass7.md](pass7.md) — design-matrix robustness on pass6's best configs
- [pass8.md](pass8.md) — RPT base pre-training (uses `_CUSTOM_BASE_CKPT` hook)
- [regime_serving.md](regime_serving.md) — serving from `regime_model_best.csv`
- [forecast_granite.md](forecast_granite.md) — serves the checkpoints
- [regime_serving.md](regime_serving.md) — reads `regime_model_best.csv`