#!/usr/bin/env python3
"""
hmm_regime_detection.py — Gaussian HMM regimes on market returns + vol.

Features (daily):
  - market equal-weight log return
  - trailing 21d realized vol
  - average pairwise correlation proxy (mean abs cross-corr sample on rolling window)

States interpreted post-hoc by mean return / vol ordering:
  low_vol, normal, high_vol_stress

Usage:
  python hmm_regime_detection.py --n-states 3 --save
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices/"
OUT_STATES = DATA_DIR / "hmm_regime_states.parquet"
OUT_SUM = DATA_DIR / "hmm_regime_summary.parquet"
OUT_TRANS = DATA_DIR / "hmm_transition_matrix.parquet"
OUT_TRIGGERS = DATA_DIR / "hmm_transition_triggers.parquet"
CKPT = DATA_DIR / "hmm_checkpoint.pkl"
FEAT_CACHE = DATA_DIR / "hmm_features_cache.parquet"

# Window policy (adaptive, not a fixed default):
#   - Fit only on the current regime episode: data since the last detected
#     transition (from the previous run's triggers file), plus a context
#     floor so EM never starves.
#   - Floor: at least 2 full regime cycles of median dwell (empirically
#     252d floor / 756d cap). A fixed 504d default is wrong: after a fresh
#     transition it dilutes the new regime with the old one; late in a
#     long regime it throws away usable history.
WINDOW_MIN = 252
WINDOW_CAP = 756
WINDOW_FALLBACK = 504  # no prior triggers file (first run)


def build_features(rets: pd.DataFrame, corr_window: int = 21) -> pd.DataFrame:
    """Vectorized HMM features: mkt_ret, vol21, avg pairwise corr.

    avg_corr computed ONLY over strictly-upper-triangular pairs (j > i):
    no self-correlations (always 1.0) and no duplicate (i,j)/(j,i) pairs.
    That halves the correlation work vs the naive full-matrix approach.
    Rolling correlations use the identity
        corr(X,Y) = cov(X,Y) / (std(X)·std(Y))
    with sums over the trailing window via cumulative sums, so the whole
    pass is O(N·k²/2) with tiny constant factor (no giant (N,k,k) arrays).
    NaN pairs (short-history tickers) are excluded per-date via nanmean.
    """
    mkt = rets.mean(axis=1)
    vol21 = mkt.rolling(21).std() * np.sqrt(252)
    k = rets.shape[1]
    if k >= 2:
        avg_corr = _pairwise_avg_corr(rets, corr_window)
    else:
        avg_corr = np.full(len(rets), np.nan)
    feat = pd.DataFrame(
        {"mkt_ret": mkt, "vol21": vol21, "avg_corr": avg_corr}, index=rets.index
    )
    return feat.dropna()


def _pairwise_avg_corr(rets: pd.DataFrame, w: int) -> np.ndarray:
    """Mean of strictly-upper-triangular rolling correlations.

    For each date t, computes corr over the trailing w-day window for every
    pair (i, j) with j > i, then averages across pairs (NaN-safe). Uses the
    sum identity so no (N, k, k) intermediate is materialized:
        Σ (x_i - μ_i)(x_j - μ_j)  =  Σ x_i x_j - (Σ x_i)(Σ x_j) / w
    The inner pair loop is JIT-compiled (numba) with strictly-upper-triangular
    indices j > i — no duplicate (i,j)/(j,i) or self pairs are computed.
    Falls back to a vectorized numpy path if numba is unavailable.
    """
    X = rets.values.astype(np.float64)
    try:
        return _pairwise_avg_corr_numba(X, w)
    except Exception:
        return _pairwise_avg_corr_np(X, w)


def _pairwise_avg_corr_numba(X: np.ndarray, w: int) -> np.ndarray:
    """numba JIT core: O(N·k²/2) over strictly-upper-triangular pairs."""
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _core(X, w):
        n, k = X.shape
        nan_mask = ~np.isnan(X)
        Xz = np.nan_to_num(X, nan=0.0)
        # prefix sums (length n+1; window (a..t] = [t+1] - [a])
        csum = np.zeros((n + 1, k))
        c2sum = np.zeros((n + 1, k))
        cnt = np.zeros((n + 1, k))
        for t in range(n):
            for c in range(k):
                v = Xz[t, c]
                csum[t + 1, c] = csum[t, c] + v
                c2sum[t + 1, c] = c2sum[t, c] + v * v
                cnt[t + 1, c] = cnt[t, c] + (1.0 if nan_mask[t, c] else 0.0)
        # per-i accumulators (parallel-safe: each i writes its own row)
        out2d = np.zeros((k, n))
        cnt2d = np.zeros((k, n))
        for i in prange(k - 1):
            for j in range(i + 1, k):  # strictly upper triangle: j > i
                run = 0.0  # sliding window sum of products
                for t in range(n):
                    p = Xz[t, i] * Xz[t, j]
                    run += p
                    if t >= w:
                        run -= Xz[t - w, i] * Xz[t - w, j]
                    if t < w - 1:
                        continue
                    a = t + 1 - w
                    if (cnt[t + 1, i] - cnt[a, i] < w) or (cnt[t + 1, j] - cnt[a, j] < w):
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
                        out2d[i, t] += cxy / denom
                        cnt2d[i, t] += 1
        res = np.full(n, np.nan)
        for t in range(n):
            s = 0.0
            c = 0
            for i in range(k):
                s += out2d[i, t]
                c += int(cnt2d[i, t])
            if c > 0:
                res[t] = s / c
        return res

    return _core(X, w)


def _pairwise_avg_corr_np(X: np.ndarray, w: int) -> np.ndarray:
    """Vectorized numpy fallback (block cumsum per i over all j > i)."""
    n, k = X.shape
    nan_mask = ~np.isnan(X)
    Xz = np.nan_to_num(X, nan=0.0)

    csum = np.vstack([np.zeros((1, k)), np.cumsum(Xz, axis=0)])
    c2sum = np.vstack([np.zeros((1, k)), np.cumsum(Xz * Xz, axis=0)])
    cnt = np.vstack([np.zeros((1, k)), np.cumsum(nan_mask.astype(np.float64), axis=0)])

    sxw_all = csum[w:] - csum[:-w]
    sx2w_all = c2sum[w:] - c2sum[:-w]
    cntw_all = cnt[w:] - cnt[:-w]
    t_idx = np.arange(w - 1, n)

    out = np.zeros(n)
    counts = np.zeros(n)
    for i in range(k - 1):
        m = k - i - 1
        prod = Xz[:, i, None] * Xz[:, i + 1:]
        cprod = np.vstack([np.zeros((1, m)), np.cumsum(prod, axis=0)])
        spw = cprod[w:] - cprod[:-w]
        sxw = sxw_all[:, i, None]
        syw = sxw_all[:, i + 1:]
        sx2w = sx2w_all[:, i, None]
        sy2w = sx2w_all[:, i + 1:]
        cxw = cntw_all[:, i, None]
        cyw = cntw_all[:, i + 1:]

        varx = (sx2w - sxw * sxw / w) / (w - 1)
        vary = (sy2w - syw * syw / w) / (w - 1)
        denom = np.sqrt(varx * vary)
        valid = (cxw >= w) & (cyw >= w) & (denom > 0)

        cov = (spw - sxw * syw / w) / (w - 1)
        contrib = np.where(valid, cov / np.where(valid, denom, 1.0), 0.0)
        cnt_contrib = valid.astype(np.float64)

        out[t_idx] += contrib.sum(axis=1)
        counts[t_idx] += cnt_contrib.sum(axis=1)

    ok = counts > 0
    res = np.full(n, np.nan)
    res[ok] = out[ok] / counts[ok]
    return res



def fit_hmm(feat: pd.DataFrame, n_states: int = 3, seed: int = 7, resume: bool = True,
            data_fp: dict | None = None):
    """Fit Gaussian HMM, warm-starting from a pickled checkpoint when possible.

    Speedup strategy (user-directed):
      - Persist the fitted model to disk (pickle) between runs.
      - Before persisting, set model.init_params = "" so a later fit() skips
        re-initialization (it continues EM from the persisted parameters).
      - On resume, only fit the NEW observations (rows after the checkpoint's
        last date); the old rows are already represented in the persisted
        parameters. This turns a ~35 min full-history fit into seconds on
        incremental days.

    Full retrain is REQUIRED when the underlying data changed structurally —
    new tickers added, history extended backwards, or history cleaned. The
    caller passes a data fingerprint (ticker set + date range); if it no
    longer matches the checkpoint's fingerprint, the checkpoint is discarded
    and the model is trained from init.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        raise SystemExit("pip install hmmlearn")

    X = feat[["mkt_ret", "vol21", "avg_corr"]].values
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    model = None
    ckpt = _load_checkpoint(n_states)
    fp_ok = _fingerprint_ok(ckpt, data_fp)
    if resume and ckpt is not None and ckpt["n_states"] == n_states and fp_ok:
        model = ckpt["model"]
        # Continue EM from persisted params — never re-initialize.
        model.init_params = ""
        ck_last = ckpt["last_date"]
        new = feat[feat.index > ck_last]
        if len(new) >= 1:
            # Fit the new rows PLUS a trailing context from the checkpoint's
            # window. With covariance_type="full" a single row (or very few)
            # makes the emission covariances singular -> 'covars must be
            # symmetric, positive-definite'. The trailing context keeps EM
            # well-posed while still warm-starting from persisted params.
            ctx = feat[feat.index <= ck_last].tail(max(60, 4 * len(new)))
            fit_rows = pd.concat([ctx, new])
            Xn = fit_rows[["mkt_ret", "vol21", "avg_corr"]].values
            # standardize with the checkpoint's saved scaler so the
            # persisted params stay comparable
            Xnz = (Xn - ckpt["mu"]) / ckpt["sd"]
            model.fit(Xnz)
            print(f"[hmm] resumed from checkpoint: warm-start fit "
                  f"{len(new)} new rows + {len(ctx)} context rows "
                  f"since {pd.Timestamp(ck_last).date()}")
        else:
            print("[hmm] checkpoint current — no new rows to fit")
    elif ckpt is not None and not fp_ok:
        print("[hmm] data fingerprint changed (tickers/history) — full retrain from init")

    if model is None:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=seed,
            tol=1e-3,
        )
        model.fit(Xz)
        print("[hmm] full fit (no usable checkpoint)")

    # Always re-predict over the FULL window with the final model.
    # Use the checkpoint's scaler if we resumed, otherwise use current.
    if ckpt is not None and model is ckpt["model"]:
        Xz = (X - ckpt["mu"]) / ckpt["sd"]
    states = model.predict(Xz)
    post = model.predict_proba(Xz)

    # Compute labels for checkpoint (needed for full transition matrix)
    labels, _ = label_states(feat, states)

    # Persist checkpoint BEFORE returning: init_params="" so the next run
    # resumes EM instead of re-initializing.
    if ckpt is not None and model is ckpt["model"]:
        mu, sd = ckpt["mu"], ckpt["sd"]
    _save_checkpoint(model, mu, sd, feat.index[-1], n_states, data_fp, labels)
    return model, states, post, mu, sd


def _load_checkpoint(n_states: int) -> dict | None:
    """Load the pickled HMM checkpoint, validating state count + model type."""
    if not CKPT.exists():
        return None
    try:
        with open(CKPT, "rb") as fh:
            ck = pickle.load(fh)
        if not isinstance(ck, dict) or ck.get("n_states") != n_states:
            return None
        from hmmlearn.hmm import GaussianHMM
        if not isinstance(ck.get("model"), GaussianHMM):
            return None
        if "last_date" not in ck or "mu" not in ck or "sd" not in ck:
            return None
        return ck
    except Exception:
        return None


def _fingerprint_ok(ckpt: dict | None, data_fp: dict | None) -> bool:
    """Checkpoint is usable iff the data it was trained on is unchanged.

    Structural key (tickers + start date) must always match — new tickers or
    backward-extended history force a full retrain. The content hash of the
    already-trained range is compared only when BOTH sides carry it (the
    first run stores no hash; a later append-only run adds one — and that
    must NOT invalidate the checkpoint).
    """
    if data_fp is None or ckpt is None:
        return True
    old = ckpt.get("data_fp") or {}
    if old.get("tickers") != data_fp.get("tickers"):
        return False
    if old.get("start") != data_fp.get("start"):
        return False
    if "hist_hash" in old and "hist_hash" in data_fp:
        return old["hist_hash"] == data_fp["hist_hash"]
    return True


def _save_checkpoint(model, mu: np.ndarray, sd: np.ndarray, last_date, n_states: int,
                     data_fp: dict | None = None, labels: dict | None = None) -> None:
    """Persist the fitted model. init_params="" is set BEFORE writing so a
    resumed fit() continues from these params instead of re-initializing."""
    model.init_params = ""
    try:
        with open(CKPT, "wb") as fh:
            pickle.dump({
                "model": model,
                "mu": mu,
                "sd": sd,
                "last_date": pd.Timestamp(last_date),
                "n_states": n_states,
                "data_fp": data_fp,
                "labels": labels,
            }, fh)
    except Exception as e:
        print(f"[hmm] WARNING: checkpoint save failed: {e}")


def data_fingerprint(wide: pd.DataFrame, ck_last=None) -> dict:
    """Fingerprint of the training data for checkpoint validity.

    Two layers:
      - structural key (tickers, start date): detects NEW tickers and
        backward-extended history. Invariant to daily appends.
      - content hash of rows at-or-before ck_last: detects cleaned/corrected
        history (values changed under the already-trained range). When
        ck_last is None only the structural key is returned (used when no
        checkpoint exists yet).

    Pure appends (new dates after ck_last on the same tickers) leave both
    layers unchanged → checkpoint resume is safe.
    """
    fp = {
        "tickers": tuple(sorted(wide.columns.astype(str))),
        "start": pd.Timestamp(wide.index[0]).date().isoformat(),
    }
    if ck_last is not None:
        old = wide[wide.index <= ck_last]
        fp["hist_hash"] = _content_hash(old)
    return fp


def _content_hash(wide: pd.DataFrame) -> str:
    """Fast deterministic hash of a price matrix (row-count invariant)."""
    import hashlib
    arr = wide.sort_index().values.astype("<f8", copy=False)
    return hashlib.sha256(np.nan_to_num(arr, nan=-999.0).tobytes()).hexdigest()


def load_features_cache(rets: pd.DataFrame, fp: dict, wide: pd.DataFrame) -> pd.DataFrame | None:
    """Return cached features if the data they were built from is unchanged.

    The cache stores {date, mkt_ret, vol21, avg_corr} plus the structural
    fingerprint and the content hash of the price matrix rows it covers.
    On a daily append the covered rows are unchanged → cache hit; only the
    new rows need fresh feature computation.
    """
    if not FEAT_CACHE.exists():
        return None
    try:
        c = pd.read_parquet(FEAT_CACHE)
        if "date" not in c.columns or "fp_tickers" not in c.columns:
            return None
        if c["fp_tickers"].iloc[0] != ",".join(fp["tickers"]):
            return None
        if c["fp_start"].iloc[0] != fp.get("start"):
            return None
        c_last = pd.Timestamp(c["date"].max())
        # hash only the rows the cache covers — unchanged on pure appends
        covered = wide[wide.index <= c_last]
        if _content_hash(covered) != c["fp_hash"].iloc[0]:
            return None
        c = c.set_index("date").sort_index()
        return c.loc[c.index.isin(rets.index)]
    except Exception:
        return None


def save_features_cache(feat: pd.DataFrame, fp: dict, wide: pd.DataFrame) -> None:
    """Persist features with the structural fingerprint + covered-row hash."""
    try:
        c = feat.reset_index().rename(columns={"index": "date"})
        c["fp_tickers"] = ",".join(fp["tickers"])
        c["fp_start"] = fp.get("start")
        covered = wide[wide.index <= c["date"].max()]
        c["fp_hash"] = _content_hash(covered)
        c.to_parquet(FEAT_CACHE, index=False)
    except Exception as e:
        print(f"[hmm] WARNING: feature cache save failed: {e}")


def label_states(feat: pd.DataFrame, states: np.ndarray) -> dict[int, str]:
    tmp = feat.copy()
    tmp["state"] = states
    g = tmp.groupby("state").agg(mean_ret=("mkt_ret", "mean"), mean_vol=("vol21", "mean"), mean_corr=("avg_corr", "mean"))
    # lowest vol -> low_vol; highest vol -> high_vol_stress; middle -> normal
    order = g.sort_values("mean_vol").index.tolist()
    labels = {}
    names = ["low_vol", "normal", "high_vol_stress"] if len(order) == 3 else [f"state_{i}" for i in range(len(order))]
    if len(order) == 2:
        names = ["low_vol", "high_vol_stress"]
    if len(order) > 3:
        names = ["low_vol"] + [f"mid_{i}" for i in range(1, len(order) - 1)] + ["high_vol_stress"]
    for i, s in enumerate(order):
        labels[int(s)] = names[i] if i < len(names) else f"state_{s}"
    return labels, g


def adaptive_window(wide: pd.DataFrame) -> int | None:
    """Pick fit window from the last regime transition (previous run's triggers).

    Returns None for full history when no prior triggers exist.
    """
    if not OUT_TRIGGERS.exists():
        return WINDOW_FALLBACK
    try:
        trig = pd.read_parquet(OUT_TRIGGERS)
        if trig.empty or "date" not in trig.columns:
            return WINDOW_FALLBACK
        last = pd.Timestamp(trig["date"].max())
        since = (wide.index[-1] - last).days
        # Fit from the transition itself, but never below the floor.
        return int(min(WINDOW_CAP, max(WINDOW_MIN, since)))
    except Exception:
        return WINDOW_FALLBACK


def run(n_states: int = 3, save: bool = True, window_days: int | None = "auto"):
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    # Fingerprint the FULL underlying data (pre-window) so structural changes
    # (new tickers, extended/cleaned history) force a full retrain while a
    # daily append keeps the fingerprint stable → checkpoint resume is safe.
    ck_last = None
    if CKPT.exists():
        try:
            with open(CKPT, "rb") as fh:
                ck_last = pickle.load(fh).get("last_date")
        except Exception:
            ck_last = None
    fp = data_fingerprint(wide, ck_last)
    if window_days == "auto":
        window_days = adaptive_window(wide)
        if window_days is not None:
            print(f"[hmm] adaptive window: {window_days} trading days (last transition policy)")
    elif window_days is not None and window_days <= 0:
        window_days = None  # explicit 0 = full history
    # Rolling window: fit only on recent history (older regimes pollute EM).
    # Regime dynamics change — the adaptive window focuses on the current episode.
    if window_days is not None and len(wide) > window_days:
        wide = wide.iloc[-window_days:]
    rets = np.log(wide / wide.shift(1)).dropna(how="all")

    # Feature cache: reuse the previous run's features when the data is
    # unchanged (same tickers/start AND same content hash over covered rows).
    # On a pure daily append the cache covers everything except the new rows
    # — only those need fresh pairwise-correlation work. Hashing the covered
    # matrix is O(N·k) and far cheaper than recomputing O(N·k²/2) correlations.
    feat = load_features_cache(rets, fp, wide)
    if feat is not None and len(feat) >= len(rets) - 21:
        print(f"[hmm] feature cache hit: {len(feat)} rows (skip pairwise corr)")
        if len(feat) < len(rets):
            # append-only: build features for the new rows only (needs the
            # trailing 21-day context for the rolling windows)
            n_new = len(rets) - len(feat)
            tail = rets.iloc[-(21 + n_new):]
            new_feat = build_features(tail).iloc[-n_new:]
            feat = pd.concat([feat, new_feat])
    else:
        feat = build_features(rets)
        print(f"[hmm] full feature build: {len(feat)} rows")
    save_features_cache(feat, fp, wide)

    model, states, post, mu, sd = fit_hmm(feat, n_states=n_states, data_fp=fp)
    labels, g = label_states(feat, states)

    # Use the FULL label set from the checkpoint (if available) or the
    # current run's labels for the transition matrix. This ensures all
    # trained states are represented even if the windowed data doesn't
    # observe every state.
    ck_labels = None
    if CKPT.exists():
        try:
            with open(CKPT, "rb") as fh:
                ck_labels = pickle.load(fh).get("labels")
        except Exception:
            ck_labels = None

    out = feat.copy()
    out["state_id"] = states
    out["regime"] = [labels[int(s)] for s in states]
    for k in range(post.shape[1]):
        out[f"p_state_{k}"] = post[:, k]

    # summary (using CURRENT run's labels, since we only summarize observed)
    rows = []
    for sid, name in labels.items():
        sub = out[out.state_id == sid]
        rows.append({
            "state_id": sid,
            "regime": name,
            "n_days": len(sub),
            "pct_time": len(sub) / len(out),
            "mean_ret_ann": float(sub.mkt_ret.mean() * 252),
            "mean_vol": float(sub.vol21.mean()),
            "mean_avg_corr": float(sub.avg_corr.mean()),
            "median_ret": float(sub.mkt_ret.median()),
        })
    summary = pd.DataFrame(rows).sort_values("mean_vol")
    print("=== HMM regime summary ===")
    print(summary.to_string(index=False))

    # transition matrix with labels — use checkpoint labels (full set) if
    # available, else current run's labels
    trans_labels = ck_labels if ck_labels is not None else labels
    state_ids = sorted(trans_labels.keys())
    labs = [trans_labels[s] for s in state_ids]
    tm = model.transmat_
    trans = pd.DataFrame(tm, index=labs, columns=labs)
    print("\n=== Transition matrix P(to|from) ===")
    print(trans.round(3).to_string())

    # dwell times
    print("\n=== Regime path (last 30 days) ===")
    print(out[["regime", "mkt_ret", "vol21", "avg_corr"]].tail(30).to_string())

    if save:
        out.reset_index().rename(columns={"index": "date"}).to_parquet(OUT_STATES, index=False)
        summary.to_parquet(OUT_SUM, index=False)
        trans.to_parquet(OUT_TRANS)
        print(f"\nWrote {OUT_STATES}\nWrote {OUT_SUM}\nWrote {OUT_TRANS}")
    return out, summary, trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=3)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--population", action="store_true")
    ap.add_argument("--adaptive-trans", action="store_true")
    ap.add_argument("--window-days", default="auto")
    args = ap.parse_args()
    if args.population:
        st = pd.read_parquet(OUT_STATES)
        st["date"] = pd.to_datetime(st["date"])
        g = st.groupby([st["date"].dt.to_period("M"), "regime"]).size().unstack(fill_value=0)
        share = g.div(g.sum(axis=1), axis=0)
        out = share.reset_index().rename(columns={"date": "month"})
        out.to_parquet(DATA_DIR / "regime_population.parquet", index=False)
        print(share.tail(6).round(3).to_string())
        print(f"months {len(share)}")
        return
    if getattr(args, "adaptive_trans", False):
        st = pd.read_parquet(OUT_STATES)
        st["date"] = pd.to_datetime(st["date"])
        st = st.sort_values("date")
        stay = (st["regime"] == st["regime"].shift(1)).astype(float)
        vol = pd.to_numeric(st["vol21"], errors="coerce")
        roll_p = stay.rolling(63).mean()
        roll_v = vol.rolling(63).mean()
        both = pd.concat([roll_p, roll_v], axis=1).dropna()
        corr = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
        out = pd.DataFrame([{"corr_persist_vol": corr, "n": int(len(both)),
                             "mean_persist": float(roll_p.dropna().mean())}])
        out.to_parquet(DATA_DIR / "adaptive_hmm_states.parquet", index=False)
        print(out.to_string(index=False))
        print(f"persist vs vol corr {corr:+.3f}  bar +0.60")
        return
    wd = "auto" if args.window_days == "auto" else int(args.window_days)
    run(n_states=args.n_states, save=True, window_days=wd)


if __name__ == "__main__":
    main()
