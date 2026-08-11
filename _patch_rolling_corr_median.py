#!/usr/bin/env python3
"""One-shot: add rolling_pairwise_stats with median (using HMM's numba core)."""
from pathlib import Path

p = Path("rolling_correlation_windows.py")
src = p.read_text(encoding="utf-8")

# Remove the old rolling_pairwise_stats and avg_pairwise functions entirely
# Find them by the signature
start = src.index("def avg_pairwise(corr: pd.DataFrame) -> tuple[float, float]:")
# Find the end of rolling_pairwise_stats (next def or main)
end_marker = "def run(windows="
end = src.index(end_marker, start)

old_block = src[start:end]

new_block = '''def _rolling_pairwise_stats_numba(X: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean AND median of strictly-upper-triangular rolling correlations.

    Extends the HMM's numba core to also collect per-date correlation lists
    for median computation. Memory: O(k²/2 * n) floats ~ 3160 pairs * 16k dates
    ~ 400MB for k=80, n=16k — acceptable.
    """
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _core(X, w):
        n, k = X.shape
        nan_mask = ~np.isnan(X)
        Xz = np.nan_to_num(X, nan=0.0)
        # prefix sums
        csum = np.zeros((X.shape[0] + 1, k))
        c2sum = np.zeros((X.shape[0] + 1, k))
        cnt = np.zeros((X.shape[0] + 1, k))
        nan_mask = ~np.isnan(X)
        Xz = np.nan_to_num(X, nan=0.0)
        for t in range(X.shape[0]):
            for c in range(k):
                v = Xz[t, c]
                csum[t + 1, c] = csum[t, c] + v
                c2sum[t + 1, c] = c2sum[t, c] + v * v
                cnt[t + 1, c] = cnt[t, c] + (1.0 if ~np.isnan(X[t, c]) else 0.0)

        # per-i accumulators: sum of corr, count, and list of corr values for median
        # We'll accumulate in per-i arrays, then merge
        out_sum = np.zeros((k, X.shape[0]))
        out_cnt = np.zeros((k, X.shape[0]))
        # For median: we need to store the individual correlation values per date
        # For memory: we'll use a 3D array (i, t) -> list of corr values
        # Instead, we collect all pair corrs per date in a 2D array (n_dates, n_pairs)
        n, k = X.shape
        n_pairs = k * (k - 1) // 2
        all_corrs = np.full((X.shape[0], n_pairs), np.nan)
        pair_idx = 0
        for i in range(k - 1):
            for j in range(i + 1, k):
                if i >= j:
                    continue
                # sliding window product sum
                run = 0.0
                csum_i = csum[:, i]
                csum_j = csum[:, j]
                c2sum_i = c2sum[:, i]
                c2sum_j = c2sum[:, j]
                cnt_i = cnt[:, i]
                cnt_j = cnt[:, j]
                run = 0.0
                for t in range(X.shape[0]):
                    p = Xz[t, i] * Xz[t, j]
                    run += p
                    if t >= w:
                        run -= Xz[t - w, i] * Xz[t - w, j]
                    if t < w - 1:
                        continue
                    a = t + 1 - w
                    if (cnt[t + 1, i] - cnt[a, i] < w) or (cnt[t + 1, j] - cnt[a, j] < w):
                        all_corrs[t, pair_idx] = np.nan
                        continue
                    ci = csum[t + 1, i] - csum[a, i]
                    cj = csum[t + 1, j] - csum[a, j]
                    c2i = c2sum[t + 1, i] - c2sum[a, i]
                    c2j = c2sum[t + 1, j] - c2sum[a, j]
                    cxy = run - ci * cj / w
                    varx = c2i - ci * ci / w
                    vary = c2j - cj * cj / w
                    denom = np.sqrt(varx * vary)
                    if denom > 0:
                        corr = cxy / denom
                        all_corrs[t, pair_idx] = corr
                pair_idx += 1
        # mean/median per date
        res_mean = np.full(X.shape[0], np.nan)
        res_med = np.full(X.shape[0], np.nan)
        for t in range(X.shape[0]):
            row = all_corrs[t, :]
            valid = row[~np.isnan(row)]
            if len(valid) > 0:
                all_corrs[t, 0] = np.nan  # placeholder
                # mean/median of valid
                res_mean[t] = np.mean(valid)
                res_med[t] = np.median(valid)
        return res_mean, res_med

    try:
        return _rolling_pairwise_stats_numba(X, w)
    except Exception:
        return _rolling_pairwise_stats_np(X, w)
'''

# This is too complex. Let me just use the simpler approach: 
# keep HMM's mean function, add a separate median path using the same numba structure