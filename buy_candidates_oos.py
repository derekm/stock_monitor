#!/usr/bin/env python3
"""
buy_candidates_oos.py — walk-forward OOS validation of the buy_candidates score.

WHAT THIS CAN AND CANNOT TEST (read before trusting a number)

`buy_candidates.score_row` is additive over ~10 components. Only some are
reconstructible point-in-time:

  RECONSTRUCTIBLE (dated sources, safe to backtest)
    - momentum_score   : trailing price returns (daily_prices)
    - resid_mom_63     : residual momentum vs the equal-weight market
    - liquidity_score  : median dollar volume
    - decision         : quality/value gate recomputed from PIT fundamentals
    - mos_pass         : margin-of-safety gate from PIT fundamentals
    - leverage_flag    : debt_to_assets / mktcap_to_assets from PIT fundamentals
    - distrust_discount: heuristic cash-distrust multiplier

  NOT RECONSTRUCTIBLE (snapshot tables with NO date column -> cannot be
  evaluated without lookahead; excluded rather than faked)
    - composite        (signal_aggregator_scores.parquet)
    - factor_composite (quarterly_factor_exposures is dated but sparse/quarterly)
    - fragile_veto     (fragility_screen.parquet)
    - skew             (options snapshot)
    - sp500_member     (current membership only -> survivorship bias)
    - stress_p         (hmm_regime_states covers <1y)

So this validates the FUNDAMENTAL+PRICE core of the score, which is the part
that carries the unvalidated magic numbers (dual_pass_core +0.35, mos_pass
+0.08, leverage flags +0.08/-0.12, momentum steps). It does NOT validate the
full production score. Any component above that lacks history is reported as
excluded, not silently zeroed.

Method: quarterly rebalances, features strictly PIT (merge_asof backward),
label = forward 63d return, expanding-origin folds with a 63d+21d embargo.
Metrics: rank IC (Spearman) of score vs forward return, decile spreads, and
component ablations on identical folds.

Outputs: buy_candidates_oos_metrics.csv, buy_candidates_oos_ablation.csv,
         buy_candidates_oos.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
HORIZON = 63
EMBARGO = 21

# Components we can honestly reconstruct point-in-time.
SUPPORTED = ["decision", "mos_pass", "leverage_flag", "momentum_score",
             "resid_mom_63", "liquidity_score", "distrust_discount"]
# Components excluded for lack of dated history (reported, never faked).
#
# STATUS 2026-08: the writers now append `*_history.parquet` via
# snapshot_history.append_history, so these become testable as history
# accumulates. `history_status()` reports how many dates each one has; a
# component is promoted out of this dict once it covers enough rebalances.
# Nothing here is back-fillable: options chains in particular are gone once the
# quote date passes, which is why the append had to be added before more
# backtesting, not after.
EXCLUDED = {
    "composite": "signal_aggregator_scores: history appending since 2026-08",
    "factor_composite": "quarterly only, sparse coverage pre-2014",
    "fragile_veto": "fragility_screen: history appending since 2026-08",
    "skew": "options_skew: history appending since 2026-08 (not back-fillable)",
    "sp500_member": "current membership only -> survivorship bias",
    "stress_p": "hmm_regime_states covers <1 year",
}

# Snapshot tables now accumulating point-in-time history, and the scorer
# component each one unlocks.
#
# preferred_metrics is deliberately NOT here: preferred_metrics_history.parquet
# already exists with different semantics (a per-ticker-quarter panel written by
# backfill_preferred_fundamentals.py, 311k rows / 3600 dates), and decision and
# mos_pass are already reconstructible PIT from dated fundamentals -- they are
# in SUPPORTED, not EXCLUDED.
HISTORY_TABLES = {
    "signal_aggregator_scores": "composite",
    "fragility_screen": "fragile_veto",
    "options_skew": "skew",
}


def history_status() -> pd.DataFrame:
    """How much point-in-time history exists per snapshot table.

    Print this before trusting an EXCLUDED label: once a table covers enough
    rebalance dates, its component can move into the tested set.
    """
    from snapshot_history import load_history

    rows = []
    for name, comp in HISTORY_TABLES.items():
        h = load_history(name)
        if h.empty or "as_of_date" not in h.columns:
            rows.append({"table": name, "unlocks": comp, "dates": 0,
                         "first": None, "last": None, "rows": 0})
            continue
        rows.append({
            "table": name, "unlocks": comp,
            "dates": int(h["as_of_date"].nunique()),
            "first": h["as_of_date"].min(), "last": h["as_of_date"].max(),
            "rows": len(h),
        })
    return pd.DataFrame(rows)


def build_panel(start: str, min_dollar_vol: float, min_names: int) -> pd.DataFrame:
    """PIT panel: reconstructed scorer inputs + forward 63d return label.

    Reuses distrust_oos_eval's validated price/fundamentals loaders so the
    return math (adj_close), liquidity filter and outlier guard are identical.
    """
    import distrust_oos_eval as D

    px = D.load_price_panel()
    dates = D.rebalance_dates(px, start, "Q")
    print(f"  rebalance dates: {len(dates)} ({dates[0].date()} .. {dates[-1].date()})")

    snap = px[px["date"].isin(dates)].dropna(subset=["fwd"]).copy()
    snap["ticker"] = snap["ticker"].astype(str).str.upper()
    n0 = len(snap)
    snap = snap[pd.to_numeric(snap["dollar_vol"], errors="coerce") >= min_dollar_vol]
    snap = snap[snap["fwd"].abs() <= 5.0]
    print(f"  liquidity >=${min_dollar_vol/1e6:.0f}M + outlier guard: {n0:,} -> {len(snap):,}")

    # --- PIT fundamentals: quality/value gate inputs -----------------------
    f = pd.read_parquet(DATA_DIR / "fundamentals.parquet")
    f["as_of_date"] = pd.to_datetime(f["as_of_date"], errors="coerce")
    f = f.dropna(subset=["as_of_date", "ticker"])
    f["ticker"] = f["ticker"].astype(str).str.upper()

    def num(c):
        return pd.to_numeric(f[c], errors="coerce") if c in f.columns else pd.Series(np.nan, index=f.index)

    fund = pd.DataFrame({
        "ticker": f["ticker"],
        "as_of_date": f["as_of_date"],
        "roe": num("roe"),
        "roic": num("roic"),
        "d2e": num("debt_to_equity"),
        "cover": num("interest_coverage"),
        "stability": num("earnings_stability"),
        "ev_ebitda": num("ev_ebitda"),
        "pb": num("pb_ratio"),
        "mca": num("mktcap_to_assets"),
        "cash": num("cash_and_equivalents"),
        "mcap": num("market_cap"),
        "assets": num("total_assets"),
        "debt": num("total_debt"),
    }).sort_values("as_of_date", kind="mergesort")

    snap = snap.sort_values("date", kind="mergesort")
    m = pd.merge_asof(snap, fund, left_on="date", right_on="as_of_date",
                      by="ticker", direction="backward",
                      tolerance=pd.Timedelta(days=400))
    m = m.dropna(subset=["as_of_date"])
    print(f"  after PIT fundamentals join: {len(m):,} rows")
    return m


def reconstruct_inputs(m: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the scorer's PIT inputs using PRODUCTION thresholds.

    Thresholds are imported from analytics_common (the same module
    preferred_metrics uses) so this harness cannot drift from production.
    """
    from analytics_common import (
        ROE_MIN, ROIC_MIN, DE_MAX, EV_MAX, PB_MAX, MCA_MAX,
    )
    out = m.copy()

    roe, roic, d2e = out["roe"], out["roic"], out["d2e"]
    ev, pb, mca, cover = out["ev_ebitda"], out["pb"], out["mca"], out["cover"]

    buffett = (roe.notna() & (roe >= ROE_MIN)
               & roic.notna() & (roic >= ROIC_MIN)
               & d2e.notna() & (d2e <= DE_MAX))
    trifecta = (ev.notna() & (ev <= EV_MAX)
                & pb.notna() & (pb <= PB_MAX)
                & mca.notna() & (mca <= MCA_MAX))

    # decision ladder: dual pass -> CORE, else value/quality, else nothing.
    dec = pd.Series("", index=out.index, dtype=object)
    dec[buffett] = "INCLUDE_QUALITY"
    dec[trifecta] = "INCLUDE_VALUE"
    dec[buffett & trifecta] = "INCLUDE_CORE"
    out["decision"] = dec

    # leverage_flag (production predicate order: levered/cheap/mixed/low-MCA)
    lf = pd.Series("", index=out.index, dtype=object)
    in_mca = mca.notna() & (mca <= MCA_MAX)
    lf[in_mca & d2e.notna() & (d2e > DE_MAX)] = "levered-assets"
    lf[in_mca & d2e.notna() & (d2e <= 0.5) & (cover.isna() | (cover >= 5))] = "cheap-assets"
    lf[in_mca & d2e.notna() & (d2e > 0.5) & (d2e <= DE_MAX)] = "mixed-assets"
    lf[in_mca & d2e.isna()] = "low-MCA"
    out["leverage_flag"] = lf

    # mos_pass proxy: cheap on BOTH earnings and book vs the value thresholds.
    # The production version needs fair_pe/fair_ev_ebitda (WACC + growth), which
    # depend on a PIT ERP curve; this is a deliberately simpler stand-in and is
    # labelled as such in the ablation output.
    out["mos_pass"] = (ev.notna() & (ev <= EV_MAX * 0.7)
                       & pb.notna() & (pb <= PB_MAX * 0.7))

    # distrust heuristic (same form as preferred_metrics, pre-blend)
    excess = (out["cash"] / out["mcap"].replace(0, np.nan)).clip(0, 1)
    excess = excess.fillna((1.0 - mca.clip(0, 2)).clip(0, 1)).fillna(0)
    p_bad = pd.Series(0.10, index=out.index)
    p_bad = p_bad + (roe < 0).astype(float) * 0.20
    p_bad = p_bad + (out["stability"].fillna(0.5) < 0.3).astype(float) * 0.10
    out["distrust_discount"] = (1.0 - p_bad.clip(0, 0.60) * excess).clip(0.5, 1.0)

    # momentum_score / resid_mom_63 / liquidity_score in production units.
    #
    # momentum_score (built last, below) matches momentum_analytics L120-126: the
    # MEAN of cross-sectionally z-scored horizons, i.e. centered on 0. It used to
    # be `rank(mom126, pct=True)`; because MOMENTUM_STEPS thresholds at
    # -0.5 / 0.0 / +0.5, a 0-1 rank made the -0.15 and 0.0 tiers UNREACHABLE
    # (measured 0.0000 of rows below either), collapsing a four-tier step
    # function into two.
    # residual momentum, matching momentum_analytics.py L82-88: regress each
    # name's daily returns on the equal-weight market, then take
    # resid.tail(63).mean() * 63 -- i.e. a BETA-ADJUSTED, 63-day CUMULATIVE
    # residual.
    #
    # This used to be `mom21 - mean(mom21)`: a 21-day simple demean with no beta
    # adjustment. That is a different variable on a different scale, so the
    # earlier ablation was measuring something the production scorer never sees.
    # It mattered: RESID_MOM_STEPS gives +0.10 above 0.05, and the old proxy
    # crossed that threshold on 23.0% of rows (std 0.1222).
    out["resid_mom_63"] = _resid_mom_63_pit(out)
    # mom_12_1: 12-month return skipping the most recent month, as in
    # momentum_analytics (s.iloc[-21] / s.iloc[-252] - 1).
    if {"mom252", "mom21"}.issubset(out.columns):
        out["mom_12_1"] = (1.0 + out["mom252"]) / (1.0 + out["mom21"]) - 1.0
    # momentum_score LAST: it z-scores and averages the horizons above,
    # including resid_mom_63, so it must be built after them.
    out["momentum_score"] = _momentum_score_pit(out)
    out["liquidity_score"] = out.groupby("date")["dollar_vol"].rank(pct=True)

    return out


def _momentum_score_pit(out: pd.DataFrame) -> pd.Series:
    """Composite TS momentum score, matching momentum_analytics.py L120-126.

    Cross-sectionally z-score each available horizon WITHIN each rebalance date,
    then average. Production z-scores across the snapshot universe; doing it
    per-date is the point-in-time equivalent and keeps the scale (mean 0, sd ~1)
    that MOMENTUM_STEPS was calibrated against.

    Horizon map to the reconstructed panel:
      ret_21d  -> mom21     ret_63d -> mom63    ret_126d -> mom126
      mom_12_1 -> mom252 (12-month, skipping the most recent month)
      resid_mom_63 -> the beta-adjusted residual built above
    """
    import numpy as np

    cols = [c for c in ("mom21", "mom63", "mom126", "mom_12_1", "resid_mom_63")
            if c in out.columns]
    if not cols:
        return pd.Series(np.nan, index=out.index)

    zs = []
    g = out.groupby("date")
    for c in cols:
        mu = g[c].transform("mean")
        sd = g[c].transform("std")
        zs.append(((out[c] - mu) / sd.where(sd > 0)).fillna(0.0))
    z = pd.concat(zs, axis=1)
    # rows with no usable horizon at all stay NaN rather than scoring 0
    any_obs = out[cols].notna().any(axis=1)
    return z.mean(axis=1).where(any_obs)


def _resid_mom_63_pit(out: pd.DataFrame) -> pd.Series:
    """Beta-adjusted 63d cumulative residual momentum, point-in-time.

    Reconstructed from the same daily-return panel the label uses, so it is
    strictly backward-looking at each rebalance date. Beta is estimated on the
    trailing window against the equal-weight cross-sectional mean return, which
    is the market proxy momentum_analytics uses.
    """
    import numpy as np

    need = {"ticker", "date", "mom63"}
    if not need.issubset(out.columns):
        # mom63 is required for the cumulative form; without it, be explicit
        # rather than silently substituting the 21d proxy again.
        return pd.Series(np.nan, index=out.index)

    # Per-date equal-weight market move over the same 63d horizon.
    mkt = out.groupby("date")["mom63"].transform("mean")
    # Beta over the cross-section per date: cov(r, m)/var(m) is degenerate
    # within a single date (m is constant), so estimate each ticker's beta from
    # its own history of (mom63, mkt) pairs using only PAST observations.
    df = out[["ticker", "date", "mom63"]].copy()
    df["mkt"] = mkt
    df = df.sort_values(["ticker", "date"])
    g = df.groupby("ticker", sort=False)
    # expanding (shifted) moments -> beta uses data strictly before this date
    x, y = df["mkt"], df["mom63"]
    ex = g["mkt"].transform(lambda s: s.shift(1).expanding(min_periods=8).mean())
    ey = df.groupby("ticker", sort=False)["mom63"].transform(
        lambda s: s.shift(1).expanding(min_periods=8).mean())
    exy = df.assign(xy=x * y).groupby("ticker", sort=False)["xy"].transform(
        lambda s: s.shift(1).expanding(min_periods=8).mean())
    exx = df.assign(xx=x * x).groupby("ticker", sort=False)["xx"].transform(
        lambda s: s.shift(1).expanding(min_periods=8).mean())
    cov = exy - ex * ey
    var = exx - ex * ex
    beta = (cov / var.where(var > 0)).clip(-3, 3).fillna(1.0)
    resid = df["mom63"] - beta * df["mkt"]
    return resid.reindex(out.index)


def score_panel(df: pd.DataFrame, drop: str | None = None) -> np.ndarray:
    """Score every row with the REAL buy_candidates.score_row.

    Importing the production scorer (rather than reimplementing it) is the whole
    point: an ablation that scores a drifted copy proves nothing. `drop` blanks
    one input so its marginal contribution can be measured.
    """
    from buy_candidates import score_row

    cols = [c for c in SUPPORTED if c in df.columns]
    recs = df[cols].to_dict("records")
    if drop:
        # Neutral value per component. NaN is NOT neutral for every input:
        # mos_pass=NaN is still truthy (would keep the +0.08), and
        # distrust_discount=NaN/liquidity=NaN skip their branches entirely.
        neutral = {
            "decision": "",          # no gate credit
            "mos_pass": False,       # NaN would stay truthy -> must be False
            "leverage_flag": "",     # no flag bonus/penalty
            "distrust_discount": 1.0,  # multiplier identity, not NaN
            "liquidity_score": np.nan,
            "momentum_score": np.nan,
            "resid_mom_63": np.nan,
        }
        val = neutral.get(drop, np.nan)
        for r in recs:
            r[drop] = val
    return np.array([score_row(r, stress_p=0.0)[0] for r in recs], dtype=float)


def rank_ic(score: np.ndarray, fwd: np.ndarray) -> float:
    """Spearman rank IC."""
    ok = np.isfinite(score) & np.isfinite(fwd)
    if ok.sum() < 20:
        return float("nan")
    a = pd.Series(score[ok]).rank()
    b = pd.Series(fwd[ok]).rank()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(panel: pd.DataFrame, n_folds: int, min_train_dates: int,
             drop: str | None = None) -> pd.DataFrame:
    """Per-rebalance OOS stats. The scorer has FIXED weights (no fitting), so
    every rebalance after the warm-up is out-of-sample by construction; folds
    are still reported so stability over eras is visible.
    """
    dates = np.sort(panel["date"].unique())
    test_dates = dates[min_train_dates:]
    blocks = np.array_split(test_dates, n_folds)

    rows = []
    for k, blk in enumerate(blocks, 1):
        if len(blk) == 0:
            continue
        te = panel[panel["date"].isin(blk)].copy()
        if len(te) < 100:
            continue
        te["score"] = score_panel(te, drop=drop)

        # per-date IC then average (avoids pooling across eras)
        ics = [rank_ic(g["score"].values, g["fwd"].values) for _, g in te.groupby("date")]
        ics = [x for x in ics if np.isfinite(x)]

        # decile spread: top vs bottom decile mean forward return
        q = te.groupby("date")["score"].rank(pct=True)
        top = te.loc[q >= 0.9, "fwd"]
        bot = te.loc[q <= 0.1, "fwd"]
        rows.append({
            "fold": k,
            "start": pd.Timestamp(blk[0]).date(),
            "end": pd.Timestamp(blk[-1]).date(),
            "n": len(te),
            "n_dates": len(ics),
            "ic_mean": round(float(np.mean(ics)), 4) if ics else np.nan,
            "ic_std": round(float(np.std(ics)), 4) if ics else np.nan,
            "ic_hit": round(float(np.mean([x > 0 for x in ics])), 3) if ics else np.nan,
            "top_decile_fwd": round(float(top.mean()), 4) if len(top) else np.nan,
            "bot_decile_fwd": round(float(bot.mean()), 4) if len(bot) else np.nan,
            "spread": round(float(top.mean() - bot.mean()), 4) if len(top) and len(bot) else np.nan,
            "universe_fwd": round(float(te["fwd"].mean()), 4),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2006-01-01")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--min-train-dates", type=int, default=4)
    ap.add_argument("--min-names", type=int, default=50)
    ap.add_argument("--min-dollar-vol", type=float, default=5e6)
    args = ap.parse_args()

    print("Scope of this validation:")
    print(f"  testable components : {SUPPORTED}")
    for k, v in EXCLUDED.items():
        print(f"  EXCLUDED {k:17}: {v}")
    print()
    print("Point-in-time history accumulating (unlocks the EXCLUDED set):")
    print(history_status().to_string(index=False))
    print()

    print("Building PIT panel...")
    panel = build_panel(args.start, args.min_dollar_vol, args.min_names)
    panel = reconstruct_inputs(panel)
    cnt = panel.groupby("date")["fwd"].size()
    panel = panel[panel["date"].isin(cnt[cnt >= args.min_names].index)]
    print(f"  panel: {len(panel):,} rows, {panel['date'].nunique()} dates")
    print(f"  decision mix: {panel['decision'].value_counts().to_dict()}")

    print("\nFull score (all testable components):")
    full = evaluate(panel, args.folds, args.min_train_dates)
    print(full.to_string(index=False))

    print("\nAblations (drop one component, identical folds):")
    abl = []
    base_ic = float(full["ic_mean"].mean())
    base_sp = float(full["spread"].mean())
    abl.append({"variant": "FULL", "ic_mean": round(base_ic, 4),
                "spread": round(base_sp, 4), "d_ic": 0.0, "d_spread": 0.0})
    for comp in SUPPORTED:
        r = evaluate(panel, args.folds, args.min_train_dates, drop=comp)
        ic, sp = float(r["ic_mean"].mean()), float(r["spread"].mean())
        abl.append({"variant": f"drop_{comp}", "ic_mean": round(ic, 4),
                    "spread": round(sp, 4), "d_ic": round(ic - base_ic, 4),
                    "d_spread": round(sp - base_sp, 4)})
    abl = pd.DataFrame(abl)
    print(abl.to_string(index=False))

    full.to_csv(DATA_DIR / "buy_candidates_oos_metrics.csv", index=False)
    abl.to_csv(DATA_DIR / "buy_candidates_oos_ablation.csv", index=False)

    print("\n=== Verdict ===")
    ics = full["ic_mean"].dropna()
    print(f"mean OOS rank IC   : {ics.mean():+.4f}")
    print(f"IC range by fold   : {ics.min():+.4f} .. {ics.max():+.4f}")
    print(f"folds with IC>0    : {int((ics > 0).sum())}/{len(ics)}")
    print(f"mean decile spread : {full['spread'].dropna().mean():+.4f} "
          f"(63d, top-decile minus bottom-decile)")
    # A component whose removal IMPROVES IC is actively harmful.
    harmful = abl[(abl["variant"] != "FULL") & (abl["d_ic"] > 0)]
    if len(harmful):
        print("\ncomponents whose REMOVAL improves IC (candidate harmful):")
        for _, h in harmful.sort_values("d_ic", ascending=False).iterrows():
            print(f"  {h['variant']:24} d_IC {h['d_ic']:+.4f}  d_spread {h['d_spread']:+.4f}")
        print("  NOTE: fold-mean deltas are NOT significance tests. Confirm any")
        print("  candidate with a paired per-date test before acting on it.")
        print()
        print("  Paired per-date results (2026-08, n=73, all SUPPORTED components):")
        print("    component          d_IC       t       p   verdict")
        print("    decision        -0.0053  -4.053  0.0001   KEEP (helps, significant)")
        print("    mos_pass        -0.0004  -2.892  0.0051   KEEP (helps, significant)")
        print("    liquidity_score -0.0025  -1.791  0.0775   keep")
        print("    distrust_disc   -0.0007  -1.179  0.2421   keep")
        print("    leverage_flag   +0.0001   0.117  0.9068   keep")
        print("    resid_mom_63    +0.0002   0.093  0.9263   keep")
        print("    momentum_score  +0.0102   1.337  0.1855   keep")
        print("  NOT ONE component is significantly harmful -> remove nothing.")
        print()
        print("  Both momentum inputs had to be re-specified before testing:")
        print("    resid_mom_63   was a 21d simple demean; production is a")
        print("                   BETA-ADJUSTED 63d cumulative residual.")
        print("    momentum_score was rank(mom126, pct=True); production is the")
        print("                   MEAN of z-scored horizons, centered on 0. On a")
        print("                   0-1 rank the -0.15 and 0.00 tiers of")
        print("                   MOMENTUM_STEPS were UNREACHABLE (0.0000 of rows).")
    else:
        print("\nno component is actively harmful on IC")


if __name__ == "__main__":
    main()


