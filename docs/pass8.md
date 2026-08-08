# pass8.py — own RPT-pre-trained base + regime fine-tunes from it

Tests whether **Resolution Prefix Tuning (RPT)** — the TTM paper's mechanism
for telling the model its sampling resolution explicitly (§3.1.1) — improves
our regime-selected forecasting, by pre-training OUR OWN base with RPT enabled
instead of fine-tuning the IBM `granite-timeseries-ttm-r2` base that was never
trained with it.

## Why it exists (rationale)

The pass6 `--rpt` probe proved the IBM base **cannot absorb RPT at fine-tune
time**: enabling `resolution_prefix_tuning` on a model pre-trained without it
adds a frequency patch (8→9 patches), which breaks the multi-level
patch-partition reshape every TSMixer layer was built for
(`mat1 384x36 vs 32x64`). RPT is a **pre-training** technique — the
resolution token must be inside the model's training distribution from step
zero, exactly as the paper pre-trains with it.

Root cause (verified by reading the tsfm source): `freq_mod` produces a
`d_model`-wide embedding concatenated onto the patch axis, so the effective
patch count is `num_patches + 1`. A model built with `num_patches=8` receives
9 patches and every layer built for 8 breaks. The fix: **build the model with
`num_patches = 9`** when RPT is enabled — then the patcher, positional
encoding, and all mixers are sized for 9 consistently and a forward with
`freq_token=2` (daily) runs clean (`(2, 96, 1)` verified).

## Pipeline

```
Stage A (--pretrain)    fresh TinyTimeMixer, resolution_prefix_tuning=True,
                        num_patches=9, freq_token=2 (daily), channel-independent
                        univariate close windows from ALL monitored tickers
                        (paper §3.1 pre-training workflow), MSE objective.
                        Saves checkpoints/rpt_base/ttm_rpt_<steps>.pt + config.
Stage B (--fine-tune)   pass6-style regime-selected sweep (same cell grid:
                        ticker × regime × steps × cap × lr, plus head-only and
                        exog toggles) but fine-tuning FROM THE RPT BASE.
                        Outputs regime_model_oos_rpt.csv / _best_rpt.csv so the
                        IBM-base and RPT-base results stay directly comparable.
Compare  (--compare)    joins the two OOS CSVs on
                        (ticker, regime, steps, cap, lr, head_only, exog) and
                        reports per-cell dir-excess delta (RPT base − IBM base).
```

The pass6 hook: `pass6._CUSTOM_BASE_CKPT` — when set, pass6's channel
expansion rebuilds the model from the custom checkpoint's saved config +
weights (loose state-dict load for channel expansion) instead of the IBM hub
model. This is how pass8's regime fine-tunes start from the RPT base.

## Experiment matrix

Stage B sweeps (per ticker × regime), all with pass6 honesty rules (shared
global boundary, 96d embargo, per-regime persistence baseline, test windows
after the boundary):

| dimension | values | question |
|---|---|---|
| base | **rpt** (ours) vs **ibm** (pass6 rows) | does an RPT-trained base beat the IBM base on OOS dir excess? |
| steps | 3000, 6000 | still the best config? |
| cap | 100, 200 | does the cap=100 finding hold? |
| lr | gd.LR, 5e-5 | lr sensitivity |
| head_only | on/off | does frozen-backbone tuning transfer to our base too? |
| exog | on/off | does the calendar-event channel stack with RPT? |

Primary metric: **dir excess over the regime's own persistence baseline**
(same as pass6) — the honest measure of forecast skill. Secondary: dir_acc at
spans h10/h21/h42/h63/h96 and MAPE.

## Honesty rules (inherited from pass6)

- GLOBAL temporal split: train targets end before a shared boundary; test
  starts after the 96d embargo; no regime's test leaks into any training.
- Regime models fine-tune from the RPT base only — no test leakage.
- Persistence baseline computed on the SAME test windows per regime.
- A regime with < MIN_TEST test windows is skipped.
- RPT freq token = 2 (daily) at every forward, training and inference;
  Stage B checkpoints record `rpt=True` and the serving side
  (forecast_granite) passes the token when loading an RPT checkpoint.

## Usage

```
# Stage A — pre-train the RPT base on the full daily history (~2-4 h GPU)
python pass8.py --pretrain --steps 8000 --save-to checkpoints/rpt_base

# Stage B — regime sweep from the RPT base (after pass6 frees the GPU)
python pass8.py --fine-tune --tickers AEP,NVR,FICO --steps 3000 6000 \
                --caps 100 200 --lrs 5e-5 --head-only --exog --resume \
                --max-experiments 120

# Compare RPT base vs IBM base on overlapping cells
python pass8.py --compare
```

## Outputs (see SCHEMAS → Forecast / anomaly family)

- `checkpoints/rpt_base/ttm_rpt_<steps>.pt` (+ `_config.json`) — Stage A base
- `regime_model_oos_rpt.csv`, `regime_model_best_rpt.csv` — Stage B results
- `rpt_vs_ibm_compare.csv` — per-cell excess delta (RPT − IBM)

## Related

pass6.py (regime sweep + `_CUSTOM_BASE_CKPT` hook + the original RPT probe),
pass7.py (design matrix — pass8 extends its arms with the `base` dimension),
forecast_granite.py (serving: rebuilds checkpoint architecture, passes
freq_token for RPT checkpoints). Granite-TTM paper §3.1.1 (RPT) and §3.2
(exogenous mixer, whose input-form we implement as the event-proximity
channel in pass6 --exog).
