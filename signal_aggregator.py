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
from pathlib import Path

import numpy as np
import pandas as pd

from analytics_common import DATA_DIR, load_adj_prices_pandas, wide_closes, clip_returns

PREF = DATA_DIR / "preferred_metrics.csv"
PEER = DATA_DIR / "peer_analytics_signals.csv"
CROSS = DATA_DIR / "cross_section_rankings.csv"
PAIR = DATA_DIR / "pair_engine_pairs.csv"
EARN = DATA_DIR / "earnings_catalyst_signals.csv"
OUT_SCORES = DATA_DIR / "signal_aggregator_scores.csv"
OUT_IC = DATA_DIR / "signal_aggregator_ic.csv"

FORWARD_HORIZON = 21


def _norm(s: pd.Series) -> pd.Series:
    """Robust [0,1] normalize by percentile rank (ties averaged)."""
    return s.rank(pct=True, na_option="keep")


def load_scores() -> pd.DataFrame:
    """One row per ticker with a raw score per family (higher = better)."""
    tickers: set[str] = set()

    def _add(df: pd.DataFrame):
        if not df.empty and "ticker" in df.columns:
            tickers.update(df["ticker"].astype(str).str.upper())

    pref = pd.read_csv(PREF) if PREF.exists() else pd.DataFrame()
    peer = pd.read_csv(PEER) if PEER.exists() else pd.DataFrame()
    cross = pd.read_csv(CROSS) if CROSS.exists() else pd.DataFrame()
    pair = pd.read_csv(PAIR) if PAIR.exists() else pd.DataFrame()
    earn = pd.read_csv(EARN) if EARN.exists() else pd.DataFrame()
    _add(pref); _add(peer); _add(cross); _add(pair); _add(earn)

    rows = []
    for tk in sorted(tickers):
        row = {"ticker": tk}

        # preferred: composite_score already [0,1]-ish; decision boosts core
        if not pref.empty:
            p = pref[pref["ticker"] == tk]
            if len(p):
                cs = p["composite_score"].iloc[0] if "composite_score" in p else np.nan
                row["preferred"] = float(cs) if pd.notna(cs) else np.nan
        # peer: best_sharpe_rank (percentile 0..1, higher = better)
        if not peer.empty:
            q = peer[peer["ticker"] == tk]
            if len(q) and "best_sharpe_rank" in q:
                row["peer"] = float(q["best_sharpe_rank"].iloc[0])
        # cross_section: bucket 1..5 → (bucket-1)/4
        if not cross.empty:
            c = cross[cross["ticker"] == tk]
            if len(c) and "bucket" in c:
                b = c["bucket"].dropna().iloc[-1]
                row["cross"] = float((b - 1) / 4) if pd.notna(b) else np.nan
        # pair_engine: z_now → |z| strength; mean-reversion favors extremes.
        # Score = min(|z|/2, 1): a spread at ±2 = full conviction.
        if not pair.empty and "z_now" in pair.columns:
            pr = pair[pair["asset_a"] == tk]
            pb = pair[pair["asset_b"] == tk]
            zs = pd.concat([pr["z_now"], pb["z_now"]]).dropna()
            if len(zs):
                row["pair"] = float(np.clip(np.abs(zs.iloc[-1]) / 2.0, 0, 1))
        # earnings: catalyst_score → scale from raw to [0,1] by rank across names
        if not earn.empty:
            e = earn[earn["ticker"] == tk]
            if len(e) and "catalyst_score" in e:
                row["earnings"] = float(e["catalyst_score"].iloc[0])

        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")


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


def estimate_ic(scores: pd.DataFrame, fwd: pd.Series) -> pd.DataFrame:
    """Trailing-window IC (rank corr of each family vs forward 21d return)."""
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
        ic_rows.append({"family": family, "ic": round(ic, 4), "n": len(common),
                        "weight": round(max(ic, 0.0), 4)})
    ic_df = pd.DataFrame(ic_rows)
    wsum = ic_df["weight"].sum()
    if wsum > 0:
        ic_df["weight_norm"] = (ic_df["weight"] / wsum).round(4)
    else:
        ic_df["weight_norm"] = 0.0
    return ic_df


def build(cutoff: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = load_scores()
    prices = load_adj_prices_pandas()
    if cutoff is None:
        cutoff = prices["date"].max()
    cutoff = pd.Timestamp(cutoff)
    fwd = forward_returns(cutoff)
    ic_df = estimate_ic(scores, fwd)

    # composite = weighted mean of normalized family scores
    normed = scores.copy()
    for family in ["preferred", "peer", "cross", "pair", "earnings"]:
        if family in normed:
            normed[family] = _norm(normed[family])
    wmap = dict(zip(ic_df["family"], ic_df["weight_norm"]))

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
    out["rank"] = out["composite"].rank(ascending=False, method="min").astype("Int64")
    out = out.sort_values("rank").reset_index()
    return out, ic_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default=None, help="YYYY-MM-DD (default: last price date)")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    scores, ic_df = build(cutoff=args.cutoff)
    print("=== Trailing-window IC (rank corr vs forward 21d return) ===")
    print(ic_df.to_string(index=False))
    print("\n=== Top 20 composite ===")
    cols = [c for c in ["ticker", "preferred", "peer", "cross", "pair", "earnings", "composite", "rank"] if c in scores]
    print(scores[cols].head(20).to_string(index=False))
    if args.save:
        scores.to_csv(OUT_SCORES, index=False)
        ic_df.to_csv(OUT_IC, index=False)
        print(f"\nWrote {OUT_SCORES}\nWrote {OUT_IC}")


if __name__ == "__main__":
    main()
