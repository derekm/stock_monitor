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

## Per-regime parameter sweep

Sweeps `steps` × window `cap` × `lr` for each regime independently, so a
regime with few windows (high_vol_stress) can get a smaller step count / cap
than a data-rich regime (low_vol, normal). Selection objective: **max OOS
direction excess over the regime's persistence baseline** — the honest
"does this model add skill in this regime" measure.

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

## Related programs

- `pass5.py` — the base honest-OOS harness (trainlast protocol)
- `regime_forecast.py` — per-regime baseline measurement that motivated this
- `hmm_regime_detection.py` — produces the regime labels consumed here
- `pass5_sweep.py` — the global param sweep this complements (per-regime)
