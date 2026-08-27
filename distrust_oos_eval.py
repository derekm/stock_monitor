#!/usr/bin/env python3
"""
distrust_oos_eval.py — honest walk-forward OOS evaluation + calibration for the
distrust fit used by preferred_metrics.py.

Why this exists: the in-script `distrust_fit_auc` splits `xm[:split]` by ROW
ORDER (alphabetical ticker) while every label comes from the SAME final 63-day
window. Train and test therefore share one identical market episode -> that
number is cross-sectional, not out-of-sample.

This module rebuilds the fit as a proper walk-forward panel:
  - labels sampled at many historical rebalance dates (not one)
  - features joined point-in-time (merge_asof, backward only)
  - expanding-origin folds with an embargo so no train label overlaps a test label
  - pooled OOS AUC + decile calibration, vs baselines

Outputs: distrust_oos_metrics.csv, distrust_oos_calibration.csv,
         distrust_oos_calibration.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices/"
FUND = DATA_DIR / "fundamentals.parquet"
ARISTA = DATA_DIR / "arista_metrics.parquet"

HORIZON = 63          # trading days, matches production P(63d ret < -10%)
DRAWDOWN = -0.10      # label threshold
EMBARGO = 21          # extra trading days purged around each test block


def _price_col(cols) -> str:
    """Prefer split/dividend-adjusted price for return math."""
    for c in ("adj_close", "close_adj", "close"):
        if c in cols:
            return c
    raise SystemExit("no usable price column in daily_prices/")


def load_price_panel() -> pd.DataFrame:
    """Long-form prices with forward label, trailing features, and liquidity."""
    head = pd.read_parquet(PRICES).head(0)
    pcol = _price_col(head.columns)
    cols = ["date", "ticker", pcol]
    for extra in ("close", "volume"):
        if extra in head.columns and extra not in cols:
            cols.append(extra)
    px = pd.read_parquet(PRICES, columns=cols)
    px["date"] = pd.to_datetime(px["date"])
    px = px.rename(columns={pcol: "px"})
    px = px.dropna(subset=["px"])
    px = px[px["px"] > 0]
    px = px.sort_values(["ticker", "date"], kind="mergesort")

    g = px.groupby("ticker", sort=False)["px"]
    # Forward return over HORIZON days -> the label (future, never a feature).
    px["fwd"] = g.shift(-HORIZON) / px["px"] - 1.0
    # Trailing features (past only).
    px["mom126"] = px["px"] / g.shift(126) - 1.0
    px["mom21"] = px["px"] / g.shift(21) - 1.0
    # 63d trailing return: needed to rebuild buy_candidates' resid_mom_63 on the
    # same horizon production uses (momentum_analytics takes a 63-day
    # beta-adjusted residual, not a 21-day demean).
    px["mom63"] = px["px"] / g.shift(63) - 1.0
    px["mom252"] = px["px"] / g.shift(252) - 1.0
    ret1 = px["px"] / g.shift(1) - 1.0
    grp = ret1.groupby(px["ticker"], sort=False)
    px["vol21"] = grp.rolling(21, min_periods=15).std().reset_index(level=0, drop=True)
    px["vol63"] = grp.rolling(63, min_periods=40).std().reset_index(level=0, drop=True)
    # Downside semi-deviation: only negative days (crash-relevant asymmetry).
    neg = ret1.where(ret1 < 0, 0.0)
    px["dvol63"] = (
        neg.groupby(px["ticker"], sort=False)
        .rolling(63, min_periods=40).std().reset_index(level=0, drop=True)
    )
    # Drawdown from trailing 252d peak (past only).
    roll_max = g.rolling(252, min_periods=60).max().reset_index(level=0, drop=True)
    px["dd252"] = (px["px"] / roll_max - 1.0).clip(-1, 0)

    # Liquidity: median 63d dollar volume (raw close x shares traded).
    if "volume" in px.columns:
        base_px = px["close"] if "close" in px.columns else px["px"]
        dollar = pd.to_numeric(base_px, errors="coerce") * pd.to_numeric(px["volume"], errors="coerce")
        px["dollar_vol"] = (
            dollar.groupby(px["ticker"], sort=False)
            .rolling(63, min_periods=30).median().reset_index(level=0, drop=True)
        )
    else:
        px["dollar_vol"] = np.nan

    print(f"  price panel: {len(px):,} rows, price col={pcol}, "
          f"dollar_vol non-null {int(px['dollar_vol'].notna().sum()):,}")
    return px



def rebalance_dates(px: pd.DataFrame, start: str, freq: str) -> pd.DatetimeIndex:
    """Quarter/month-end trading dates that have a fully observable forward label."""
    all_d = np.sort(px["date"].unique())
    # Last date with an observable HORIZON-day forward return.
    if len(all_d) <= HORIZON:
        raise SystemExit("not enough price history")
    usable_end = all_d[-(HORIZON + 1)]
    cand = pd.Series(all_d, index=pd.DatetimeIndex(all_d))
    grp = cand.index.to_period(freq)
    last_per = cand.groupby(grp).max()
    out = pd.DatetimeIndex(last_per.values)
    out = out[(out >= pd.Timestamp(start)) & (out <= usable_end)]
    return out


def load_fundamentals_pit() -> pd.DataFrame:
    """Point-in-time fundamentals: excess cash share + quality/decline proxies."""
    f = pd.read_parquet(FUND)
    f["as_of_date"] = pd.to_datetime(f["as_of_date"], errors="coerce")
    f = f.dropna(subset=["as_of_date", "ticker"])

    num = lambda c: pd.to_numeric(f[c], errors="coerce") if c in f.columns else pd.Series(np.nan, index=f.index)

    cash, mc = num("cash_and_equivalents"), num("market_cap")
    excess = (cash / mc.replace(0, np.nan)).clip(0, 1)
    # Same fallback ladder as preferred_metrics when cash/mktcap is missing.
    m2a = num("mktcap_to_assets")
    excess = excess.fillna((1.0 - m2a.clip(0, 2)).clip(0, 1))

    out = pd.DataFrame({
        "ticker": f["ticker"].astype(str).str.upper(),
        "as_of_date": f["as_of_date"],
        "excess": excess,
        "roe": num("roe"),
        "earnings_stability": num("earnings_stability"),
        "d2e": num("debt_to_equity"),
    })
    out = out.dropna(subset=["as_of_date"]).sort_values("as_of_date", kind="mergesort")
    print(f"  fundamentals: {len(out):,} rows, excess non-null {int(out['excess'].notna().sum()):,}")
    return out


def load_arista_flags() -> set[str]:
    if not ARISTA.exists():
        return set()
    ar = pd.read_parquet(ARISTA)
    if "ticker" not in ar.columns:
        return set()
    return set(ar["ticker"].astype(str).str.upper())


def build_panel(start: str, freq: str, min_names: int,
                min_dollar_vol: float = 0.0, max_fwd: float = 5.0) -> pd.DataFrame:
    """Assemble (date, ticker, features, label) with strict PIT joins.

    min_dollar_vol filters microcaps by median 63d dollar volume (applied at
    each rebalance date, using only trailing data). max_fwd drops absurd
    forward returns from bad split/adjustment data.
    """
    px = load_price_panel()
    dates = rebalance_dates(px, start, freq)
    print(f"  rebalance dates: {len(dates)} ({dates[0].date()} .. {dates[-1].date()})")

    snap = px[px["date"].isin(dates)].copy()
    snap = snap.dropna(subset=["fwd"])
    snap["ticker"] = snap["ticker"].astype(str).str.upper()

    n_pre = len(snap)
    if min_dollar_vol > 0:
        snap = snap[pd.to_numeric(snap["dollar_vol"], errors="coerce") >= min_dollar_vol]
        print(f"  liquidity filter >=${min_dollar_vol/1e6:.1f}M/day: "
              f"{n_pre:,} -> {len(snap):,} rows")
    if max_fwd is not None:
        n_b = len(snap)
        snap = snap[snap["fwd"].abs() <= max_fwd]
        if n_b != len(snap):
            print(f"  outlier guard |fwd|<={max_fwd}: {n_b:,} -> {len(snap):,} rows")

    fund = load_fundamentals_pit()
    snap = snap.sort_values("date", kind="mergesort")
    # backward asof => only fundamentals already published at that date
    merged = pd.merge_asof(
        snap, fund,
        left_on="date", right_on="as_of_date",
        by="ticker", direction="backward",
        tolerance=pd.Timedelta(days=400),
    )
    merged = merged.dropna(subset=["excess"])

    flags = load_arista_flags()
    merged["arista"] = merged["ticker"].isin(flags).astype(float)

    # Decline / low-quality proxies (guarded: require a positive baseline so a
    # negative ROE isn't auto-flagged as deteriorating).
    merged["lowq"] = (1.0 - merged["earnings_stability"].fillna(0.5).clip(0, 1)).clip(0, 1)
    merged["decline"] = (
        (merged["roe"] < 0) | (merged["mom126"].fillna(0) < -0.20)
    ).astype(float)

    merged["y"] = (merged["fwd"] < DRAWDOWN).astype(float)

    keep = ["date", "ticker", "y", "excess", "decline", "arista", "lowq",
            "mom21", "mom126", "mom252", "vol21", "vol63", "dvol63", "dd252",
            "dollar_vol", "fwd"]
    keep = [c for c in keep if c in merged.columns]
    panel = merged[keep].dropna(subset=["excess", "lowq"]).copy()

    # Drop thin dates that can't support a cross-section.
    cnt = panel.groupby("date")["y"].size()
    good = cnt[cnt >= min_names].index
    panel = panel[panel["date"].isin(good)]
    print(f"  panel: {len(panel):,} rows, {panel['date'].nunique()} dates, "
          f"base rate {panel['y'].mean():.3f}")
    return panel.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)



FEATURE_SETS = {
    # Production fit as-is.
    "production": ["excess", "decline", "arista", "lowq"],
    # Pure price/risk features (no fundamentals join at all).
    "price": ["vol63", "dvol63", "mom126", "dd252"],
    # Price/risk + the fundamentals overlay: does the overlay add anything?
    "hybrid": ["vol63", "dvol63", "mom126", "dd252", "excess", "lowq"],
    # Single-feature reference.
    "vol_only": ["vol63"],
}
FEATURES = FEATURE_SETS["production"]

# Features ranked cross-sectionally within each date before fitting. Puts every
# feature on a comparable 0-1 scale per date and neutralizes market-wide level
# shifts (a vol spike that lifts every name carries no cross-sectional signal).
RANK_WITHIN_DATE = True


def design(df: pd.DataFrame, features: list[str] | None = None) -> np.ndarray:
    feats = features if features is not None else FEATURES
    cols = [np.ones(len(df))]
    for f in feats:
        v = pd.to_numeric(df[f], errors="coerce")
        if RANK_WITHIN_DATE and "date" in df.columns and df["date"].nunique() > 1:
            v = v.groupby(df["date"]).rank(pct=True)
        cols.append(v.fillna(0.5 if RANK_WITHIN_DATE else 0.0).values)
    return np.column_stack(cols)



def fit_logit(X: np.ndarray, y: np.ndarray, iters: int = 12, ridge: float = 1e-6) -> np.ndarray:
    """Newton-Raphson logit — same solver shape as preferred_metrics."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ b, -20, 20)
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1 - p) + 1e-6
        grad = X.T @ (p - y) + ridge * b
        hess = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        b = b - step
        if not np.all(np.isfinite(b)):
            return np.zeros(X.shape[1])
    return b


def predict(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(X @ b, -20, 20)))


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC (ties averaged). NaN when only one class present."""
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def walk_forward(panel: pd.DataFrame, n_folds: int, min_train_dates: int,
                 features: list[str] | None = None) -> pd.DataFrame:
    """Expanding-origin folds over REBALANCE DATES with a label embargo.

    A training date is usable only if its label window closed before the test
    block starts: train_date + HORIZON + EMBARGO <= test_start.
    """
    dates = np.sort(panel["date"].unique())
    if len(dates) < min_train_dates + n_folds:
        raise SystemExit(f"only {len(dates)} usable dates; need >= {min_train_dates + n_folds}")

    test_dates = dates[min_train_dates:]
    blocks = np.array_split(test_dates, n_folds)

    rows = []
    for k, blk in enumerate(blocks, 1):
        if len(blk) == 0:
            continue
        test_start = blk[0]
        # embargo in calendar days ~ (HORIZON + EMBARGO) trading days
        cutoff = pd.Timestamp(test_start) - pd.Timedelta(days=int((HORIZON + EMBARGO) * 1.45))
        tr = panel[panel["date"] <= cutoff]
        te = panel[panel["date"].isin(blk)]
        if len(tr) < 200 or len(te) < 50 or tr["y"].nunique() < 2 or te["y"].nunique() < 2:
            continue

        b = fit_logit(design(tr, features), tr["y"].values)
        p = predict(design(te, features), b)

        rows.append(pd.DataFrame({
            "fold": k,
            "date": te["date"].values,
            "ticker": te["ticker"].values,
            "y": te["y"].values,
            "p": p,
            "fwd": te["fwd"].values,
            # baselines evaluated on the SAME rows
            "p_excess": pd.to_numeric(te["excess"], errors="coerce").fillna(0).values,
            "p_lowq": pd.to_numeric(te["lowq"], errors="coerce").fillna(0).values,
            "p_vol": pd.to_numeric(te["vol21"], errors="coerce").fillna(0).values,
            "train_rows": len(tr),
            "train_end": cutoff,
        }))
        print(f"  fold {k}: train<={cutoff.date()} ({len(tr):,} rows) "
              f"test {pd.Timestamp(blk[0]).date()}..{pd.Timestamp(blk[-1]).date()} ({len(te):,} rows) "
              f"AUC={auc_score(te['y'].values, p):.3f}")

    if not rows:
        raise SystemExit("no valid folds produced — check panel density")
    return pd.concat(rows, ignore_index=True)


def calibration_table(oos: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Decile calibration: predicted vs realized drawdown frequency."""
    q = pd.qcut(oos["p"].rank(method="first"), bins, labels=False, duplicates="drop")
    g = oos.assign(bin=q).groupby("bin")
    tab = pd.DataFrame({
        "n": g.size(),
        "p_mean": g["p"].mean(),
        "p_lo": g["p"].min(),
        "p_hi": g["p"].max(),
        "realized": g["y"].mean(),
        "mean_fwd": g["fwd"].mean(),
    }).reset_index()
    # Wilson 95% interval on the realized rate
    z, n, ph = 1.96, tab["n"], tab["realized"]
    den = 1 + z**2 / n
    ctr = (ph + z**2 / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    tab["lo95"] = (ctr - half).clip(0, 1)
    tab["hi95"] = (ctr + half).clip(0, 1)
    return tab


def metrics(oos: pd.DataFrame) -> pd.DataFrame:
    """Pooled + per-fold AUC, Brier, and baseline comparisons."""
    def brier(y, p):
        return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))

    base = float(oos["y"].mean())
    rows = [{
        "scope": "pooled",
        "n": len(oos),
        "base_rate": round(base, 4),
        "auc_model": round(auc_score(oos["y"], oos["p"]), 4),
        "auc_excess_only": round(auc_score(oos["y"], oos["p_excess"]), 4),
        "auc_lowq_only": round(auc_score(oos["y"], oos["p_lowq"]), 4),
        "auc_vol21": round(auc_score(oos["y"], oos["p_vol"]), 4),
        "brier_model": round(brier(oos["y"], oos["p"]), 5),
        "brier_base": round(brier(oos["y"], np.full(len(oos), base)), 5),
    }]
    for k, g in oos.groupby("fold"):
        rows.append({
            "scope": f"fold{int(k)}",
            "n": len(g),
            "base_rate": round(float(g["y"].mean()), 4),
            "auc_model": round(auc_score(g["y"], g["p"]), 4),
            "auc_excess_only": round(auc_score(g["y"], g["p_excess"]), 4),
            "auc_lowq_only": round(auc_score(g["y"], g["p_lowq"]), 4),
            "auc_vol21": round(auc_score(g["y"], g["p_vol"]), 4),
            "brier_model": round(brier(g["y"], g["p"]), 5),
            "brier_base": round(brier(g["y"], np.full(len(g), float(g["y"].mean()))), 5),
        })
    return pd.DataFrame(rows)


def plot_calibration(tab: pd.DataFrame, met: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = met[met["scope"] == "pooled"].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))

    a = ax[0]
    a.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect calibration")
    a.errorbar(tab["p_mean"], tab["realized"],
               yerr=[tab["realized"] - tab["lo95"], tab["hi95"] - tab["realized"]],
               fmt="o-", color="#c0392b", capsize=3, lw=1.6, label="OOS deciles")
    a.axhline(pooled["base_rate"], color="#2980b9", ls=":", lw=1.3,
              label=f"base rate {pooled['base_rate']:.3f}")
    lim = max(tab["p_mean"].max(), tab["realized"].max(), pooled["base_rate"]) * 1.25
    a.set_xlim(0, lim); a.set_ylim(0, lim)
    a.set_xlabel("predicted P(63d return < -10%)")
    a.set_ylabel("realized frequency")
    a.set_title(f"OOS calibration — walk-forward\nAUC={pooled['auc_model']:.3f} "
                f"(n={int(pooled['n']):,})")
    a.legend(fontsize=8); a.grid(alpha=.25)

    b = ax[1]
    fo = met[met["scope"] != "pooled"]
    b.bar(fo["scope"], fo["auc_model"], color="#c0392b", label="model")
    b.plot(fo["scope"], fo["auc_vol21"], "o--", color="#f39c12", label="vol21 baseline")
    b.axhline(0.5, color="#888", ls="--", lw=1, label="coin flip")
    b.axhline(0.65, color="#27ae60", ls=":", lw=1.3, label="0.65 gate")
    b.axhline(pooled["auc_model"], color="#2c3e50", ls="-", lw=1, alpha=.6, label="pooled")
    b.set_ylim(0, 1); b.set_ylabel("OOS AUC"); b.set_title("AUC stability by fold")
    b.legend(fontsize=8); b.grid(alpha=.25, axis="y")
    b.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  wrote {path.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2006-01-01", help="first rebalance date")
    ap.add_argument("--freq", default="Q", choices=["Q", "M"], help="rebalance frequency")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--min-train-dates", type=int, default=12)
    ap.add_argument("--min-names", type=int, default=50)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--min-dollar-vol", type=float, default=0.0,
                    help="min median 63d dollar volume, e.g. 5e6")
    ap.add_argument("--max-fwd", type=float, default=5.0,
                    help="drop |forward return| above this (bad adjustment data)")
    ap.add_argument("--feature-set", default="production",
                    choices=sorted(FEATURE_SETS), help="feature set for the headline run")
    ap.add_argument("--compare-all", action="store_true",
                    help="evaluate every feature set on identical folds")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    print("Building point-in-time panel...")
    panel = build_panel(args.start, args.freq, args.min_names,
                        min_dollar_vol=args.min_dollar_vol, max_fwd=args.max_fwd)

    sets = sorted(FEATURE_SETS) if args.compare_all else [args.feature_set]
    results = {}
    for name in sets:
        print(f"\nWalk-forward folds (embargoed) — feature set '{name}': "
              f"{FEATURE_SETS[name]}")
        oos = walk_forward(panel, args.folds, args.min_train_dates, FEATURE_SETS[name])
        results[name] = (oos, metrics(oos))

    if args.compare_all:
        comp = []
        for name, (_, met) in results.items():
            pooled = met[met["scope"] == "pooled"].iloc[0]
            folds = met[met["scope"] != "pooled"]["auc_model"].dropna()
            comp.append({
                "feature_set": name,
                "n_features": len(FEATURE_SETS[name]),
                "auc_pooled": pooled["auc_model"],
                "auc_min_fold": round(float(folds.min()), 4),
                "auc_max_fold": round(float(folds.max()), 4),
                "brier": pooled["brier_model"],
                "brier_base": pooled["brier_base"],
                "auc_vol21_ref": pooled["auc_vol21"],
            })
        comp = pd.DataFrame(comp).sort_values("auc_pooled", ascending=False)
        comp.to_csv(DATA_DIR / f"distrust_oos_feature_sets{args.tag}.csv", index=False)
        print("\n=== Feature-set comparison (identical folds) ===")
        print(comp.to_string(index=False))
        best = comp.iloc[0]["feature_set"]
        print(f"\nbest feature set: {best}")
    else:
        best = args.feature_set

    oos, met = results[best]
    tab = calibration_table(oos, args.bins)

    met.to_csv(DATA_DIR / f"distrust_oos_metrics{args.tag}.csv", index=False)
    tab.to_csv(DATA_DIR / f"distrust_oos_calibration{args.tag}.csv", index=False)
    plot_calibration(tab, met, DATA_DIR / f"distrust_oos_calibration{args.tag}.png")

    print(f"\n=== OOS metrics ({best}) ===")
    print(met.to_string(index=False))
    print("\n=== Calibration deciles ===")
    print(tab.round(4).to_string(index=False))

    pooled = met[met["scope"] == "pooled"].iloc[0]
    folds = met[met["scope"] != "pooled"]["auc_model"].dropna()
    print("\n=== Verdict ===")
    print(f"feature set         : {best} {FEATURE_SETS[best]}")
    print(f"pooled OOS AUC      : {pooled['auc_model']:.3f}")
    print(f"fold AUC range      : {folds.min():.3f} .. {folds.max():.3f}")
    print(f"best single feature : {max(pooled['auc_excess_only'], pooled['auc_lowq_only']):.3f}")
    print(f"vol21 baseline      : {pooled['auc_vol21']:.3f}")
    print(f"Brier model/base    : {pooled['brier_model']:.5f} / {pooled['brier_base']:.5f}")
    gate = 0.65
    passes = pooled["auc_model"] >= gate and folds.min() >= 0.55
    print(f"gate AUC>={gate} and all folds>=0.55 : {'PASS' if passes else 'FAIL'}")


if __name__ == "__main__":
    main()



