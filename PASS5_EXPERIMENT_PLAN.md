# pass-5 Systematic Experiment Plan

## Goal
Find a training regime (train window, stride, steps, LR, pretrained) that produces **good OOS forecasts** (MAPE < persistence, direction > 55%) on the preceding 10y holdout.

## Current Baseline (from pass5.py trainlast mode, 6000 steps)
| Ticker | Stride | CAP | MAPE% | Dir% | Pers MAPE% | Pers Dir% |
|--------|--------|-----|-------|------|------------|-----------|
| AEP    | fixed200 | 200 | ~8.3 | ~67 | ~5.9 | ~33 |
| NVR    | fixed200 | 200 | ~8.7 | ~61 | ~8.0 | ~33 |
| FICO   | fixed200 | 200 | ~16.6 | ~58 | ~10.4 | ~36 |

**Key finding:** Direction beats persistence (~60% vs ~34%), but **MAPE loses** to persistence on all 3.

## Experimental Levers

### 1. Train Window Length (TRAIN_LEN)
Current: 10y (2520 days)
- Try: 5y (1260), 7y (1764), 15y (3780), 20y (5040)
- Hypothesis: Shorter = more recent/relevant; Longer = more patterns but more stale

### 2. Test Window Length (TEST_LEN)  
Current: 10y (2520 days)
- Try: 5y (1260), 15y (3780)
- Keep test ≥ 5y for statistical power

### 3. Stride & CAP (window sampling)
Current: P2_WIN = {fixed200: stride=1 cap=200, scaled400: stride=1 cap=400, half_wstride256: stride=256, quarter_wstride128: stride=128}
- **stride=1, cap=200**: dense recent windows (fixed200) — best so far
- Try: stride=1 cap=800, stride=64 cap=400, stride=256 cap=2000
- Tradeoff: more windows = more gradient updates per epoch, but correlated

### 4. Training Steps
Current: 6000
- Try: 1500, 3000, 9000, 12000
- Early stopping if loss plateaus

### 5. Learning Rate
pass4 uses `gd.LR` — check value and try 1e-4, 5e-4, 1e-3

### 6. Pretrained Checkpoint
Current: `pretrained=False` (IBM base only) — no full-history contamination
- Try `pretrained=True` (warm start from full-history checkpoint)
- Risk: leakage if test period influenced the warm checkpoint

### 8. Data Preprocessing
- Adj-close (current) vs raw close
- Log returns vs raw prices (TTM expects raw prices)
- Per-ticker normalization (z-score per context window)

### 9. Model Configuration
- Freeze backbone, train head only vs full fine-tune
- Context length (CONTEXT=512) / Horizon (HORIZON=96) — fixed by TTM

### 10. Multiple Tickers
Current: AEP, NVR, FICO (defensive, high-vol, financial)
- Add: KO, XOM, JPM, JNJ, PEP, CAT
- Test if regime-specific tuning helps

## Recommended Experiment Matrix (prioritized)

### Tier 1: Core variations (run first)
| Train | Test | Stride | CAP | Steps | Pretrained |
|-------|------|--------|-----|-------|------------|
| 5y    | 10y  | 1      | 200 | 6000  | False |
| 10y   | 10y  | 1      | 200 | 6000  | False |
| 15y   | 10y  | 1      | 200 | 6000  | False |
| 10y   | 10y  | 1      | 400 | 6000  | False |
| 10y   | 10y  | 64     | 200 | 6000  | False |
| 10y   | 10y  | 1      | 200 | 3000  | False |
| 10y   | 10y  | 1      | 200 | 9000  | False |
| 10y   | 10y  | 1      | 200 | 6000  | True  |

### Tier 2: Multi-ticker validation (after Tier 1 finds best config)
- Run best config on: KO, XOM, JPM, JNJ, PEP, CAT, DUK, SO
- Check if defensive vs cyclical need different configs

### Tier 3: Advanced
- Per-ticker normalization (z-score context)
- Frozen backbone + train head only
- LR sweep: 1e-4, 5e-4, 1e-3, 2e-3
- Ensemble: average predictions from 5 seeds

## Success Criteria
**Primary:** MAPE < persistence MAPE on test holdout  
**Secondary:** Direction accuracy > 55% (currently ~60% — already good)  
**Tertiary:** MAE improvement, calibration

## How to Run Locally
```bash
cd stock_monitor
# Quick test (1 config)
python pass5.py --tickers AEP --steps 1500 --strides fixed200

# Full Tier 1 sweep (edit sweep params in pass5_sweep.py --quick)
python pass5_sweep.py --quick --output /tmp/pass5_tier1.jsonl

# Full sweep (all 7 tickers, all configs)
python pass5_sweep.py --output /tmp/pass5_full.jsonl
```

## Analysis Script
```python
import json, pandas as pd
df = pd.read_json("/tmp/pass5_sweep.jsonl", lines=True)
# Filter successful
df = df[~df.get("skipped", False)]
# Beat persistence?
df["beats_pers_mape"] = df["mape"] < df["pers_mape"]
df["beats_pers_dir"] = df["dir_acc"] > df["pers_dir"]
# Best configs
best = df[df["beats_pers_mape"]].sort_values("mape")
print(best[["ticker","mode","train_window_years","stride","cap","steps","mape","dir_acc"]].head(20))
```

## Next Steps for You
1. Install missing deps in venv: `pip install polars tsfm-public`
2. Run Tier 1 matrix (≈50 experiments, ~30 min on GPU)
3. Analyze results, pick top 3 configs
4. Run Tier 2 on 7 tickers
3. If nothing beats persistence on MAPE → try per-ticker normalization (Tier 3)

## Hypothesis
The 10y train window may be too long (includes stale regimes). **5-7y train window** might capture recent regime better and beat persistence on MAPE. The direction signal is already strong (~60%) — the gap is level accuracy.