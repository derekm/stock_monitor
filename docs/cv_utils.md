# cv_utils.py

Overfitting guards shared by `pair_engine.py`, `earnings_catalyst.py`, and `cross_section.py`.

## Why it exists (rationale)

Every stat in the sprint engines must be OOS. This module centralizes the three
guards so no engine can silently report an in-sample number:

1. **Purged K-fold CV + embargo** — `purged_folds()` / `embargo_mask()`: train and
   test windows are temporally disjoint with a 21-day embargo on either side of
   the boundary (past leaks forward via autocorrelation; future leaks backward
   via label lookahead).
2. **Benjamini-Hochberg FDR** — `bh_fdr()`: controls the false-discovery rate
   across the many simultaneous tests (hundreds of cointegration pairs, dozens
   of drift buckets). Default alpha 0.10.
3. **OOS stats vs baseline** — `oos_stats_vs_baseline()`: the only numbers worth
   quoting — model vs persistence/equal-weight on the *same* OOS days.

## Usage

```python
from cv_utils import purged_folds, bh_fdr, oos_stats_vs_baseline

folds = purged_folds(n=2000, n_folds=5, embargo=21)   # [(train_idx, test_idx), ...]
sig = bh_fdr(pvals, alpha=0.10)                        # bool mask, True = survives
stats = oos_stats_vs_baseline(model_rets, baseline_rets)
```

## Outputs

None (library only). Imported by the three sprint engines.

## Related programs

- `pair_engine.py` — FDR on cointegration p-values; walk-forward via fold boundaries
- `earnings_catalyst.py` — drift buckets are trailing-window only (same discipline)
- `cross_section.py` — OOS stats vs equal-weight long baseline
- `pass5.py` — the original honest-OOS harness this pattern derives from
