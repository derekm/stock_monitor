# pass5_sweep.md — full 648-experiment sweep results

## Why it exists (rationale)

pass5 established Granite-TTM is a direction forecaster (beats persistence on
direction, loses on MAPE). The sweep systematically searches the training
parameter space (train-window length, window stride/density, window cap,
training steps) to find what actually moves OOS direction accuracy — and what
doesn't. Completed 2026-08 as 648 experiments (3 tickers × 216 configs each).

## Results (full 648-experiment space, /tmp/pass5_tier1.jsonl)

### Overall
- **mean dir 54.3% vs mean persistence 30.4%** — beat rate **75.8%**
- All three tickers beat persistence on mean direction:
  AEP 60.8% (pers 39.1%, +21.7pt) · FICO 58.1% (pers 26.5%, +31.6pt) ·
  NVR 43.9% (pers 25.7%, +18.2pt)

### Parameter effects (mean OOS dir over all configs)

| Parameter | Effect | Reading |
|---|---|---|
| **stride** | 1: 57.2% · 128: 52.3% · 256: 53.3% | **dense windows (stride=1) train best on average** — consistent with pass7's cap=100 finding |
| **steps** | 1500: 53.9% · 3000: 53.7% · 6000: 55.3% | flat (≤1.6pt spread) — more training buys nothing for direction |
| **cap** | 100: 54.1% · 200: 53.9% · 400: 54.8% | flat — window count cap is not a driver |
| **train window** | 5y: 53.8% · 10y: 55.2% · 15y: 52.2% · 20y: 55.9% | 10-20y train windows modestly better than 15y dip; wide spread small |

Note: `max dir = 100%` rows at stride 256/cap 400 are degenerate configs with
very few test windows (n_test small) — the mean is the honest number.

## Practical conclusions

1. **Direction skill is real and robust**: 3/3 tickers beat their persistence
   baseline across the parameter space (75.8% of configs).
2. **Dense windows > sparse windows** (stride=1 mean 57.2% vs 52-53% at
   sparse strides) — training-window density is the one parameter that
   consistently moves direction accuracy.
3. **Training steps are a non-factor** (flat 53.7-55.3%) — saves GPU hours.
4. This informed pass6/pass7: cap=100 dense-window regime models and the
   steps ∈ {3000, 6000} grid.

## Related programs

- `pass5.py` — the honest-OOS harness each experiment runs
- `pass6.py` / `pass7.py` — the regime-selected passes built on these findings
- `regime_forecast.py` — the regime-conditioned evaluation
