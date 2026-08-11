#!/usr/bin/env python3
"""One-shot: replace per-sample block.corr() loop with single rolling().corr() per window."""
from pathlib import Path

p = Path("rolling_correlation_windows.py")
src = p.read_text(encoding="utf-8")

old_start = '''def rolling_pairwise_stats(rets: pd.DataFrame, w: int, sample_idx: np.ndarray):'''
old_end = '''    # market vol: rolling std of the EW market return over the window
    mkt = rets.mean(axis=1).values
    mkt_vol = np.full(n_samp, np.nan)
    for ti, t in enumerate(tpos):
        blk = mkt[t + 1 - w : t + 1]
        mkt_vol[ti] = float(np.std(blk) * np.sqrt(252))
    return rets.index[tpos], avg, med, mkt_vol'''

i0 = src.index(old_start)
i1 = src.index(old_end) + len(old_end)

new_fn = '''def rolling_pairwise_stats(rets: pd.DataFrame, w: int, sample_idx: np.ndarray):
    """Mean/median pairwise corr + mkt vol over a trailing w-day window.

    Uses ONE pandas rolling().corr() pass per window (C-optimized, all dates
    at once) instead of ~3,250 block.corr() calls. The stacked result is a
    (date, ticker) MultiIndex frame; for each sampled date we take the k×k
    corr matrix of that row's series, extract the strictly-upper-triangle,
    and average. Matches block.corr() exactly (same pandas corr semantics).
    Returns (dates, avg_arr, med_arr, vol_arr).
    """
    tpos = np.asarray(sample_idx, dtype=np.int64)
    n_samp = len(tpos)
    # single rolling corr pass: MultiIndex (date, ticker), values = corr of
    # that ticker's window with every column (k values per (date,ticker))
    rc = rets.rolling(w).corr()
    k = rets.shape[1]
    cols = list(rets.columns)
    idx_map = {c: i for i, c in enumerate(cols)}
    avg = np.full(n_samp, np.nan)
    med = np.full(n_samp, np.nan)
    dates = rets.index[tpos]
    # group the stacked frame by date
    rc2 = rc.reset_index()
    rc2.columns = ["date", "ticker", *cols]
    for ti, t in enumerate(tpos):
        d = dates[ti]
        sub = rc2[rc2["date"] == d]
        if sub.empty or len(sub) < k:
            continue
        sub = sub.set_index("ticker").reindex(cols)
        vals = sub[cols].to_numpy(dtype=float)
        i_idx, j_idx = np.triu_indices(k, k=1)
        tri = vals[i_idx, j_idx]
        tri = tri[np.isfinite(tri)]
        if len(tri):
            avg[ti] = float(np.mean(tri))
            med[ti] = float(np.median(tri))
    # market vol: rolling std of the EW market return over the window
    mkt = rets.mean(axis=1).values
    mkt_vol = np.full(n_samp, np.nan)
    for ti, t in enumerate(tpos):
        blk = mkt[t + 1 - w : t + 1]
        mkt_vol[ti] = float(np.std(blk) * np.sqrt(252))
    return dates, avg, med, mkt_vol'''

src = src[:i0] + new_fn + src[i1:]
p.write_text(src, encoding="utf-8")
print("patched: single rolling().corr() per window")
