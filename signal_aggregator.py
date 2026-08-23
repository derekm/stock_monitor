#!/usr/bin/env python3
"""
signal_aggregator.py — Combine the five signal families into one per-ticker
composite with OOS-derived weights.

Signal families consumed:
  preferred_metrics (decision, composite_score)      — single-name dual screen
  peer_analytics (peer_signal, best_sharpe_rank)     — relative vs group
  cross_section (bucket 1..5)                        — cross-sectional rank
  pair_engine (z_now)                                — relative-value spread z
  earnings_catalyst (catalyst_score)                 — event timing

Method (all OOS):
  1. Normalize each family to a per-ticker score on [0,1] (higher = better).
  2. Estimate each family's information coefficient (rank corr of score vs
     forward 21d return) on a TRAILING window only (--ic-window days ending
     at --cutoff). Weight = max(IC, 0) (negative-IC family contributes 0).
  3. Composite = weighted mean of normalized scores (weights sum to 1 over
     positive-IC families).
  4. Report the IC table (the only numbers worth quoting) + live composite.

Outputs:
  signal_aggregator_scores.csv   ticker, per-family scores, composite, rank
  signal_aggregator_ic.csv       family, ic (trailing-window OOS)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_common import DATA_DIR, load_adj_prices_pandas, wide_closes, clip_returns

PREF = DATA_DIR / "preferred_metrics.parquet"
PEER = DATA_DIR / "peer_analytics_signals.parquet"
CROSS = DATA_DIR / "cross_section_rankings.parquet"
PAIR = DATA_DIR / "pair_engine_pairs.parquet"
EARN = DATA_DIR / "earnings_catalyst_signals.parquet"
OUT_SCORES = DATA_DIR / "signal_aggregator_scores.parquet"
OUT_IC = DATA_DIR / "signal_aggregator_ic.parquet"
OUT_W_DYN = DATA_DIR / "signal_weights_dynamic.parquet"
DECAY_PATH = DATA_DIR / "signal_decay_params.json"

FORWARD_HORIZON = 21

# Factor-adjusted residual scores (optional enhancement)
FACTOR_RESIDUALS = DATA_DIR / "signal_residual_scores.parquet"


def _norm(s: pd.Series) -> pd.Series:
    """Robust [0,1] normalize by percentile rank (ties averaged)."""
    return s.rank(pct=True, na_option="keep")


def load_scores(use_residuals: bool = False) -> pd.DataFrame:
    """One row per ticker with a raw score per family (higher = better).
    Vectorized: single pass merges instead of per-ticker filtering.

    If use_residuals=True, use factor-adjusted residual scores instead of
    raw family scores (for factor-neutral signal aggregation).
    """
    # Load factor-adjusted residuals if requested
    residuals = None
    if use_residuals and FACTOR_RESIDUALS.exists():
        residuals = pd.read_parquet(FACTOR_RESIDUALS)

    pref = pd.read_parquet(PREF) if PREF.exists() else pd.DataFrame()
    peer = pd.read_parquet(PEER) if PEER.exists() else pd.DataFrame()
    cross = pd.read_parquet(CROSS) if CROSS.exists() else pd.DataFrame()
    pair = pd.read_parquet(PAIR) if PAIR.exists() else pd.DataFrame()
    earn = pd.read_parquet(EARN) if EARN.exists() else pd.DataFrame()

    tickers: set[str] = set()
    for df in (pref, peer, cross, pair, earn):
        if not df.empty and "ticker" in df.columns:
            tickers.update(df["ticker"].astype(str).str.upper())

    # Start with all tickers as index
    idx = pd.Index(sorted(tickers), name="ticker")
    out = pd.DataFrame(index=idx)

    # Helper to get score (raw or residual)
    def get_score(df, col_name, fam_name):
        if df.empty or col_name not in df.columns:
            return pd.Series(dtype=float, index=idx)
        d = df[["ticker", col_name]].copy()
        d["ticker"] = d["ticker"].astype(str).str.upper()
        d = d.groupby("ticker")[col_name].last()
        s = d.reindex(idx)
        return s

    # preferred: composite_score or preferred_residual
    if use_residuals and residuals is not None and "preferred_residual" in residuals.columns:
        r = residuals[["ticker", "preferred_residual"]].copy()
        r["ticker"] = r["ticker"].astype(str).str.upper()
        r = r.groupby("ticker")["preferred_residual"].last()
        out["preferred"] = r.reindex(idx)
    else:
        out["preferred"] = get_score(pref, "composite_score", "preferred")

    # peer: best_sharpe_rank or peer_residual
    if use_residuals and residuals is not None and "peer_residual" in residuals.columns:
        r = residuals[["ticker", "peer_residual"]].copy()
        r["ticker"] = r["ticker"].astype(str).str.upper()
        r = r.groupby("ticker")["peer_residual"].last()
        out["peer"] = r.reindex(idx)
    else:
        out["peer"] = get_score(peer, "best_sharpe_rank", "peer")

    # cross_section: bucket 1..5 → (bucket-1)/4 or cross_residual
    if use_residuals and residuals is not None and "cross_residual" in residuals.columns:
        r = residuals[["ticker", "cross_residual"]].copy()
        r["ticker"] = r["ticker"].astype(str).str.upper()
        r = r.groupby("ticker")["cross_residual"].last()
        out["cross"] = r.reindex(idx)
    else:
        if not cross.empty and "bucket" in cross.columns:
            c = cross[["ticker", "bucket"]].copy()
            c["ticker"] = c["ticker"].astype(str).str.upper()
            c = c.groupby("ticker")["bucket"].last()
            out["cross"] = ((c - 1) / 4).reindex(idx)

    # pair_engine: z_now → min(|z|/2, 1) (no residual for pair yet)
    if not pair.empty and "z_now" in pair.columns:
        pa = pair[["asset_a", "z_now"]].rename(columns={"asset_a": "ticker"})
        pb = pair[["asset_b", "z_now"]].rename(columns={"asset_b": "ticker"})
        p = pd.concat([pa, pb])
        p["ticker"] = p["ticker"].astype(str).str.upper()
        p = p.groupby("ticker")["z_now"].last()
        out["pair"] = np.clip(np.abs(p.reindex(idx)) / 2.0, 0, 1)

    # earnings: catalyst_score or earnings_residual
    if use_residuals and residuals is not None and "earnings_residual" in residuals.columns:
        r = residuals[["ticker", "earnings_residual"]].copy()
        r["ticker"] = r["ticker"].astype(str).str.upper()
        r = r.groupby("ticker")["earnings_residual"].last()
        out["earnings"] = r.reindex(idx)
    else:
        out["earnings"] = get_score(earn, "catalyst_score", "earnings")

    return out


def forward_returns(cutoff: pd.Timestamp) -> pd.Series:
    """Forward 21d return for each ticker, measured at cutoff - horizon.

    IC weights must be estimated on data that ENDS before the live point —
    using returns that only become observable 21d later would leak the future.
    So the IC measurement date is cutoff - FORWARD_HORIZON (fully observable),
    and the composite is applied to the LIVE snapshot at cutoff.
    """
    prices = load_adj_prices_pandas()
    wide = wide_closes(prices).sort_index()
    meas_date = cutoff - pd.Timedelta(days=FORWARD_HORIZON * 2)  # ~1 month of calendar
    wide_m = wide[wide.index <= meas_date]
    if len(wide_m) < FORWARD_HORIZON + 2:
        return pd.Series(dtype=float)
    fwd = wide_m.shift(-FORWARD_HORIZON) / wide_m - 1.0
    obs = fwd.dropna(how="all")
    if len(obs) == 0:
        return pd.Series(dtype=float)
    return obs.iloc[-1]


def load_regime_map() -> pd.Series:
    """Date-indexed HMM regime labels (hmm_regime_states.parquet), or empty."""
    hmm = DATA_DIR / "hmm_regime_states.parquet"
    if not hmm.exists():
        return pd.Series(dtype=str)
    df = pd.read_parquet(hmm)
    if "date" not in df.columns or "regime" not in df.columns:
        return pd.Series(dtype=str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.set_index("date")["regime"].astype(str)


def forward_return_series(cutoff: pd.Timestamp, trailing: int = 504) -> pd.DataFrame:
    """Forward 21d returns for each trailing date (cols = tickers, rows = dates).

    Only dates with fully-observable forward returns (<= cutoff - horizon) are
    kept — no future leak. This is the per-date cross-section needed for
    regime-conditioned IC estimation.
    """
    prices = load_adj_prices_pandas()
    wide = wide_closes(prices).sort_index()
    wide = wide[wide.index <= cutoff]
    if len(wide) < FORWARD_HORIZON + 2:
        return pd.DataFrame()
    fwd = wide.shift(-FORWARD_HORIZON) / wide - 1.0
    fwd = fwd.dropna(how="all")
    if len(fwd) > trailing:
        fwd = fwd.tail(trailing)
    return fwd


def estimate_ic(scores: pd.DataFrame, fwd: pd.Series, lindy: bool = False) -> pd.DataFrame:
    """Trailing-window IC (rank corr of each family vs forward 21d return).

    lindy=True applies Taleb's Lindy factor: weight = max(IC,0) * survival
    bonus, where the bonus grows with the family's observation history
    (age of the signal is its own robustness evidence). Without lindy the
    weight is pure max(IC,0) — a 2-year-old signal and a 20-year-old signal
    with equal IC get equal weight.
    """
    ic_rows = []
    for family in ["preferred", "peer", "cross", "pair", "earnings"]:
        if family not in scores:
            continue
        s = scores[family].dropna()
        common = s.index.intersection(fwd.dropna().index)
        if len(common) < 20:
            ic_rows.append({"family": family, "ic": np.nan, "n": 0, "weight": 0.0})
            continue
        ic = float(s.loc[common].corr(fwd.loc[common], method="spearman"))
        lindy_factor = (len(common) / 504) ** 0.5 if lindy else 1.0  # ~2y history = 1.0
        ic_rows.append({"family": family, "ic": round(ic, 4), "n": len(common),
                        "weight": round(max(ic, 0.0) * lindy_factor, 4)})
    ic_df = pd.DataFrame(ic_rows)
    wsum = ic_df["weight"].sum()
    if wsum > 0:
        ic_df["weight_norm"] = (ic_df["weight"] / wsum).round(4)
    else:
        ic_df["weight_norm"] = 0.0
    return ic_df


def estimate_ic_by_regime(scores: pd.DataFrame, fwd_series: pd.DataFrame,
                          regime_s: pd.Series, lindy: bool = False) -> pd.DataFrame:
    """Per-regime IC: mean rank corr of each family vs forward returns, by the
    HMM regime in force at each measurement date.

    Current-regime IC is the honest weight for the live snapshot: if the
    signal only predicts in calm regimes, it should not get full weight now.
    Falls back to the global IC when a regime has < 20 observations.
    """
    if fwd_series.empty or regime_s.empty:
        return estimate_ic(scores, fwd_series.iloc[-1] if len(fwd_series) else pd.Series(dtype=float), lindy)

    # tag each measurement date with the regime in force at-or-before it
    tags: dict[pd.Timestamp, str] = {}
    dates = sorted(fwd_series.index)
    rdates = regime_s.index
    for d in dates:
        prior = regime_s[rdates <= d]
        tags[d] = str(prior.iloc[-1]) if len(prior) else "unknown"

    rows = []
    for family in ["preferred", "peer", "cross", "pair", "earnings"]:
        if family not in scores:
            continue
        s = scores[family].dropna()
        if len(s) < 20:
            rows.append({"family": family, "ic": np.nan, "n": 0, "weight": 0.0})
            continue
        per_regime: dict[str, list[float]] = {}
        for d in dates:
            if d not in tags:
                continue
            fwd_row = fwd_series.loc[d].dropna()
            common = s.index.intersection(fwd_row.index)
            if len(common) < 20:
                continue
            ic = s.loc[common].corr(fwd_row.loc[common], method="spearman")
            if np.isfinite(ic):
                per_regime.setdefault(tags[d], []).append(ic)
        row = {"family": family}
        for reg in ["low_vol", "normal", "high_vol_stress"]:
            vals = per_regime.get(reg, [])
            row[f"ic_{reg}"] = round(float(np.mean(vals)), 4) if len(vals) >= 5 else np.nan
            row[f"n_{reg}"] = len(vals)
        # weight from the CURRENT regime's IC (else global mean of per-regime)
        cur = tags.get(dates[-1], "unknown") if dates else "unknown"
        if cur in per_regime and len(per_regime[cur]) >= 5:
            ic_now = float(np.mean(per_regime[cur]))
        else:
            all_ics = [v for vals in per_regime.values() for v in vals]
            ic_now = float(np.mean(all_ics)) if all_ics else np.nan
        row["ic"] = round(ic_now, 4) if np.isfinite(ic_now) else np.nan
        row["regime_now"] = cur
        row["n"] = int(sum(len(v) for v in per_regime.values()))
        lindy_factor = (row["n"] / 504) ** 0.5 if lindy else 1.0
        row["weight"] = round(max(ic_now, 0.0) * lindy_factor, 4) if np.isfinite(ic_now) else 0.0
        rows.append(row)
    ic_df = pd.DataFrame(rows)
    wsum = ic_df["weight"].sum()
    if wsum > 0:
        ic_df["weight_norm"] = (ic_df["weight"] / wsum).round(4)
    else:
        ic_df["weight_norm"] = 0.0
    return ic_df


def load_decay_params() -> dict:
    if DECAY_PATH.exists():
        return json.loads(DECAY_PATH.read_text(encoding="utf-8"))
    return {
        "preferred": {"half_life_days": 126},
        "peer": {"half_life_days": 63},
        "cross": {"half_life_days": 21},
        "pair": {"half_life_days": 10},
        "earnings": {"half_life_days": 5},
    }


def regime_confidence() -> float:
    """Current HMM max posterior, else 1.0."""
    hmm = DATA_DIR / "hmm_regime_states.parquet"
    if not hmm.exists():
        return 1.0
    df = pd.read_parquet(hmm)
    pcols = [c for c in df.columns if str(c).startswith("p_state")]
    if not pcols:
        return 1.0
    last = df.sort_values("date").iloc[-1] if "date" in df.columns else df.iloc[-1]
    return float(max(float(last[c]) for c in pcols))


def apply_pedersen_weights(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Weight ∝ max(IC,0) / (turnover × cost) × regime-confidence × decay.

    Turnover = 252 / half_life. Cost = cost_model.ROUND_TRIP_BPS.
    Decay = 0.5 ** (FORWARD_HORIZON / half_life) — fast families fade at the 21d IC horizon.
    """
    from cost_model import ROUND_TRIP_BPS
    decay = load_decay_params()
    conf = regime_confidence()
    cost = max(ROUND_TRIP_BPS, 1.0) / 1e4
    out = ic_df.copy()
    hl, to, dec = [], [], []
    for fam in out["family"]:
        h = float(decay.get(str(fam), {}).get("half_life_days", 21))
        hl.append(h)
        to.append(252.0 / h)
        dec.append(0.5 ** (FORWARD_HORIZON / h))
    out["half_life_days"] = hl
    out["turnover_ann"] = to
    out["decay"] = dec
    out["regime_conf"] = conf
    ic_pos = out["ic"].clip(lower=0).fillna(0.0)
    raw = ic_pos * out["decay"] * conf / (out["turnover_ann"] * cost)
    out["weight_dyn"] = raw
    wsum = float(out["weight_dyn"].sum())
    out["weight_dyn_norm"] = (out["weight_dyn"] / wsum).round(4) if wsum > 0 else 0.0
    return out


def build(cutoff: pd.Timestamp | None = None, lindy: bool = False, use_residuals: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = load_scores(use_residuals=use_residuals)
    # Load prices once, derive both cutoff and forward series
    prices = load_adj_prices_pandas()
    if cutoff is None:
        cutoff = prices["date"].max()
    cutoff = pd.Timestamp(cutoff)
    # Record the resolved cutoff so callers can stamp history with the DATA
    # date rather than the wall-clock date (they differ on a stale run).
    build.last_cutoff = cutoff
    regime_s = load_regime_map()
    fwd_series = forward_return_series(cutoff)
    if not fwd_series.empty and not regime_s.empty:
        ic_df = estimate_ic_by_regime(scores, fwd_series, regime_s, lindy)
        print(f"Per-regime IC weights (regime_now={ic_df['regime_now'].iloc[0] if len(ic_df) else '?'})"
              + (" + Lindy survival factor" if lindy else ""))
    else:
        fwd = forward_returns(cutoff)
        ic_df = estimate_ic(scores, fwd, lindy)
        print("Global IC weights (no HMM regime map available)" + (" + Lindy" if lindy else ""))

    # composite = weighted mean of normalized family scores
    normed = scores.copy()
    for family in ["preferred", "peer", "cross", "pair", "earnings"]:
        if family in normed:
            normed[family] = _norm(normed[family])
    wmap = dict(zip(ic_df["family"], ic_df["weight_norm"]))

    # Vectorized composite calculation
    comp = pd.Series(0.0, index=normed.index)
    wsum = 0.0
    for family, w in wmap.items():
        if w > 0 and family in normed:
            comp = comp + w * normed[family].fillna(0.5)  # neutral fill for missing
            wsum += w
    if wsum > 0:
        comp = comp / wsum
    else:
        comp = pd.Series(np.nan, index=normed.index)

    out = normed.copy()
    out["composite"] = comp.round(4)
    try:
        pref = pd.read_parquet(PREF, columns=["ticker", "life_cycle_stage"])
        pref["ticker"] = pref["ticker"].astype(str).str.upper()
        lc = pref.drop_duplicates("ticker").set_index("ticker")["life_cycle_stage"]
        stage = out.index.map(lc)
        tilt = pd.Series(1.0, index=out.index)
        tilt = tilt.mask(pd.Series(stage, index=out.index).isin(["Young Growth", "High Growth"]), 1.05)
        tilt = tilt.mask(pd.Series(stage, index=out.index).eq("Decline"), 0.90)
        out["composite"] = (out["composite"] * tilt).clip(0, 1).round(4)
        out["life_cycle_stage"] = stage
    except Exception:
        pass
    out["rank"] = out["composite"].rank(ascending=False, method="min").astype("Int64")
    out = out.sort_values("rank").reset_index()
    return out, ic_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default=None, help="YYYY-MM-DD (default: last price date)")
    ap.add_argument("--lindy", action="store_true",
                    help="Lindy-weight the IC: scale each family weight by its survival "
                         "history (older signals get more weight — Taleb)")
    ap.add_argument("--use-residuals", action="store_true",
                    help="Use factor-adjusted residual scores (factor-neutral)")
    ap.add_argument("--dynamic", action="store_true",
                    help="Pedersen weights: IC / (turnover×cost) × decay × regime-conf")
    ap.add_argument("--from-ic", action="store_true",
                    help="Apply --dynamic to existing signal_aggregator_ic.parquet (no prices)")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    if args.from_ic:
        if not OUT_IC.exists():
            raise SystemExit("missing signal_aggregator_ic.parquet")
        ic_df = apply_pedersen_weights(pd.read_parquet(OUT_IC))
        print("=== Pedersen dynamic weights (from stored IC) ===")
        print(ic_df.to_string(index=False))
        if args.save:
            ic_df.to_parquet(OUT_W_DYN, index=False)
            print(f"Wrote {OUT_W_DYN}")
        return

    scores, ic_df = build(cutoff=args.cutoff, lindy=args.lindy, use_residuals=args.use_residuals)
    if args.dynamic:
        ic_df = apply_pedersen_weights(ic_df)
    print("=== Trailing-window IC (rank corr vs forward 21d return) ===")
    print(ic_df.to_string(index=False))
    print("\n=== Top 20 composite ===")
    cols = [c for c in ["ticker", "preferred", "peer", "cross", "pair", "earnings", "composite", "rank"] if c in scores]
    print(scores[cols].head(20).to_string(index=False))
    if args.save:
        scores.to_parquet(OUT_SCORES)
        ic_df.to_parquet(OUT_IC)
        print(f"\nWrote {OUT_SCORES}\nWrote {OUT_IC}")
        if args.dynamic:
            ic_df.to_parquet(OUT_W_DYN, index=False)
            print(f"Wrote {OUT_W_DYN}")
        # Append point-in-time history. The snapshot above is overwritten every
        # run, which is why buy_candidates_oos could not backtest `composite`
        # (no date column -> no way to know what the score saw historically).
        # History is additive; nothing downstream changes.
        from snapshot_history import append_history
        append_history(scores, "signal_aggregator_scores",
                       as_of=getattr(build, "last_cutoff", None) or args.cutoff)


if __name__ == "__main__":
    main()