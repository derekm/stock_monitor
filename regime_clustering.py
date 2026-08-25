#!/usr/bin/env python3
"""regime_clustering.py — Lopez de Prado regime clustering (Phase 1.5 deliverable).

Plan item (docs/RESEARCH_INTEGRATION_PLAN.md 1.5):
    "Regime clustering: replace HMM in hmm_regime_detection.py with Lopez de
     Prado's Hierarchical Risk Parity + regime clustering (codependence +
     distance correlation) -> regime_clusters.parquet"
    Success metric: "regime clusters reduce within-cluster correlation
     dispersion by >= 20%"

WHAT THIS ACTUALLY TESTS (and why it is not a literal "replace")
---------------------------------------------------------------
`hmm_regime_detection.py` labels DATES by market-level features (mkt_ret,
vol21, avg_corr). It answers "what regime is the market in today". HRP /
distance-correlation clustering groups ASSETS by codependence. Those are
different objects, so clustering cannot "replace" the HMM date labeller
without losing the date->regime series that pass6/pass8/regime_serving
consume. What the success metric actually describes is an ASSET grouping:
dispersion of pairwise correlation WITHIN a group.

So the measurable claim is: do codependence clusters group assets more tightly
(by correlation dispersion) than the incumbent grouping, which is GICS sector?
That is a real, falsifiable A/B with a stated bar, and it is what gets
measured here. The HMM date labeller is left in place.

Codependence follows Lopez de Prado (AFML ch.4 / "Codependence"):
    - correlation distance:  d_ij = sqrt(0.5 * (1 - rho_ij))       (a true metric)
    - distance correlation:  Szekely's dCor, which is 0 iff independent and
      therefore catches NON-LINEAR codependence that Pearson rho misses.
Linkage is single/average/ward on the distance matrix; the HRP quasi-diagonal
seriation is used to order the leaves (Lopez de Prado's getQuasiDiag).

Usage:
  python regime_clustering.py --save
  python regime_clustering.py --metric dcor --k 11 --save
  python regime_clustering.py --years 5 --min-cov 0.9 --save
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
OUT_CLUSTERS = DATA_DIR / "regime_clusters.parquet"
OUT_DISPERSION = DATA_DIR / "regime_cluster_dispersion.parquet"
OUT_SWEEP = DATA_DIR / "regime_cluster_sweep.parquet"


# ── codependence ────────────────────────────────────────────────────────────
def corr_distance(corr: np.ndarray) -> np.ndarray:
    """Lopez de Prado correlation distance: d = sqrt(0.5*(1-rho)).

    A proper metric (satisfies triangle inequality), unlike 1-rho.
    """
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d


def distance_correlation_matrix(X: np.ndarray, max_n: int = 750,
                                seed: int = 7) -> np.ndarray:
    """Szekely distance correlation between every pair of columns of X.

    dCor is 0 iff the variables are independent, so it detects non-linear
    codependence that Pearson rho reports as ~0. Cost is O(n^2) per pair in
    the sample dimension, so the sample is subsampled to `max_n` rows
    (deterministically) to keep a k^2 pair sweep tractable.
    """
    n_obs, k = X.shape
    if n_obs > max_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_obs, size=max_n, replace=False))
        X = X[idx]
        n_obs = max_n

    # Precompute each column's double-centred distance matrix once (the
    # expensive part), then every pair is a cheap elementwise product.
    centred = []
    dvar = np.empty(k)
    for j in range(k):
        v = X[:, j].astype(np.float64)
        a = np.abs(v[:, None] - v[None, :])
        a -= a.mean(axis=0, keepdims=True)
        a -= a.mean(axis=1, keepdims=True)
        a += a.mean()
        centred.append(a)
        dvar[j] = np.sqrt(max((a * a).mean(), 0.0))

    out = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            denom = dvar[i] * dvar[j]
            if denom <= 0:
                r = 0.0
            else:
                dcov2 = (centred[i] * centred[j]).mean()
                r = float(np.sqrt(max(dcov2, 0.0) / denom))
            out[i, j] = out[j, i] = min(r, 1.0)
    return out


# ── HRP seriation (Lopez de Prado getQuasiDiag) ─────────────────────────────
def quasi_diagonalize(link: np.ndarray, n_leaves: int) -> list[int]:
    """Reorder leaves so similar assets sit adjacent (HRP quasi-diagonal)."""
    link = link.astype(int)
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while max(order) >= n_leaves:
        out = []
        for item in order:
            if item < n_leaves:
                out.append(item)
            else:
                row = link[item - n_leaves]
                out.extend([int(row[0]), int(row[1])])
        order = out
    return order


# ── dispersion metric (the actual bar) ──────────────────────────────────────
def within_group_dispersion(corr: pd.DataFrame, labels: pd.Series) -> dict:
    """Std-dev of within-group pairwise correlations, size-weighted.

    This is the quantity the plan's >=20% reduction bar refers to. Groups with
    fewer than 2 members contribute no pairs and are excluded (they would
    otherwise flatter the score by contributing zero dispersion).
    """
    tick = [t for t in corr.index if t in labels.index]
    disp, sizes, means, singles = [], [], [], 0
    for g, members in labels.loc[tick].groupby(labels.loc[tick]).groups.items():
        m = [t for t in members if t in corr.index]
        if len(m) < 2:
            singles += 1
            continue
        sub = corr.loc[m, m].to_numpy()
        iu = np.triu_indices(len(m), k=1)
        vals = sub[iu]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        disp.append(float(np.std(vals)))
        means.append(float(np.mean(vals)))
        sizes.append(vals.size)
    if not disp:
        return {"dispersion": float("nan"), "n_groups": 0, "n_pairs": 0,
                "mean_within_corr": float("nan"), "singletons": singles}
    w = np.asarray(sizes, dtype=float)
    return {
        "dispersion": float(np.average(disp, weights=w)),
        "n_groups": len(disp),
        "n_pairs": int(w.sum()),
        "mean_within_corr": float(np.average(means, weights=w)),
        "singletons": singles,
    }


# ── data ────────────────────────────────────────────────────────────────────
def load_returns(years: float = 5.0, min_cov: float = 0.95,
                 max_assets: int | None = None) -> pd.DataFrame:
    """Daily returns panel for liquid, well-covered names.

    Reuses the Bogle TMI liquidity gate so the clustered universe is the
    investable one, not 16k tickers of which most are untradeable OTC.
    """
    import pyarrow.parquet as pq

    tbl = pq.read_table(PRICES_FILE, columns=["ticker", "date", "close"])
    df = pd.DataFrame({
        "ticker": tbl.column("ticker").to_pandas().astype(str).str.upper(),
        "date": pd.to_datetime(tbl.column("date").to_pandas()),
        "close": tbl.column("close").to_pandas().astype("float64"),
    })
    del tbl
    cutoff = df["date"].max() - pd.Timedelta(days=int(years * 365.25))
    df = df[df["date"] >= cutoff]
    wide = df.pivot_table(index="date", columns="ticker", values="close").sort_index()

    # drop holiday/thin dates the same way build_bogle_funds does
    n = wide.notna().sum(axis=1)
    wide = wide.loc[n >= max(50, float(n.median()) * 0.25)]

    cov = wide.notna().mean()
    wide = wide.loc[:, cov >= min_cov]

    # liquid, exchange-listed common only (same gate family as TMI)
    try:
        ms = pd.read_parquet(STOCKS_FILE, columns=["ticker", "instrument_type", "exchange"])
        ms["ticker"] = ms["ticker"].astype(str).str.upper()
        listed = set(ms.loc[ms["instrument_type"].eq("stock")
                            & ms["exchange"].astype(str).isin({"NMS", "NYQ", "NCM", "NGM", "ASE"}),
                            "ticker"])
        keep = [c for c in wide.columns if c in listed]
        if keep:
            wide = wide[keep]
    except Exception as e:
        print(f"  WARNING exchange filter skipped: {e}")

    # pct_change makes the FIRST row all-NaN, so dropna(axis=1, how="any")
    # deleted every column (0 assets survived). Drop that row first, then keep
    # columns by coverage and fill the residual gaps with 0 return — a missing
    # print is "no move", and correlation needs an aligned matrix.
    rets = wide.pct_change().replace([np.inf, -np.inf], np.nan).iloc[1:]
    keep = rets.notna().mean() >= min_cov
    rets = rets.loc[:, keep].fillna(0.0)
    rets = rets.loc[:, rets.std() > 0]  # constant columns have undefined corr
    if max_assets and rets.shape[1] > max_assets:
        # Keep the most-traded names (median dollar volume), not an alphabetical
        # slice: clustering the A-names would be a biased sample of the market.
        # Deterministic tie-break by ticker so runs are reproducible.
        try:
            import pyarrow.parquet as pq
            vt = pq.read_table(PRICES_FILE, columns=["ticker", "date", "close", "volume"])
            vd = pd.DataFrame({
                "ticker": vt.column("ticker").to_pandas().astype(str).str.upper(),
                "date": pd.to_datetime(vt.column("date").to_pandas()),
                "dollar": vt.column("close").to_pandas().astype("float64")
                          * vt.column("volume").to_pandas().astype("float64"),
            })
            del vt
            vd = vd[vd["ticker"].isin(set(rets.columns)) & (vd["date"] >= cutoff)]
            adv = vd.groupby("ticker")["dollar"].median()
            ranked = sorted(rets.columns, key=lambda t: (-float(adv.get(t, 0.0)), t))
            rets = rets[ranked[:max_assets]]
        except Exception as e:
            print(f"  WARNING dollar-volume ranking failed ({e}); using alphabetical slice")
            rets = rets[sorted(rets.columns)[:max_assets]]
    print(f"  returns panel: {rets.shape[0]} dates x {rets.shape[1]} tickers")
    return rets


def sector_labels(tickers: list[str]) -> pd.Series:
    ms = pd.read_parquet(STOCKS_FILE, columns=["ticker", "sector"])
    ms["ticker"] = ms["ticker"].astype(str).str.upper()
    s = ms.drop_duplicates("ticker").set_index("ticker")["sector"]
    s = s[s.apply(lambda x: isinstance(x, str) and x.strip() != "")]
    return s.reindex([t for t in tickers if t in s.index])


def _name_clusters(clusters: pd.DataFrame) -> dict[int, str]:
    """Name each cluster by its dominant sector composition.

    The name is "<dominant_sector>_<purity_pct>" so downstream consumers can
    see at a glance what the cluster represents and how pure it is. Clusters
    where no sector dominates (>50%) are named "mixed_<topsector>".
    """
    names = {}
    for c, g in clusters.groupby("cluster"):
        top = g["sector"].value_counts()
        if len(top) == 0:
            names[c] = f"cluster_{c}"
            continue
        sector = top.index[0]
        purity = top.iloc[0] / len(g)
        if purity >= 0.5:
            names[c] = f"{sector.lower().replace(' ', '_')}_{int(purity*100):02d}"
        else:
            names[c] = f"mixed_{sector.lower().replace(' ', '_')}"
    return names


# ── main ────────────────────────────────────────────────────────────────────
def run(metric: str = "corr", k: int | None = None, years: float = 5.0,
        min_cov: float = 0.95, linkage_method: str = "average",
        max_assets: int | None = None, save: bool = False) -> dict:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    print(f"Regime clustering (metric={metric}, linkage={linkage_method})")
    rets = load_returns(years=years, min_cov=min_cov, max_assets=max_assets)
    if rets.shape[1] < 10:
        raise ValueError(f"too few assets to cluster: {rets.shape[1]}")

    pear = rets.corr()
    tickers = list(pear.columns)

    if metric == "dcor":
        print("  computing distance correlation (non-linear codependence)...")
        cod = distance_correlation_matrix(rets.to_numpy())
        cod = pd.DataFrame(cod, index=tickers, columns=tickers)
        # dCor is in [0,1] with 1 = fully dependent, so 1-dCor is the distance
        dist = np.sqrt(np.clip(1.0 - cod.to_numpy(), 0.0, None))
        np.fill_diagonal(dist, 0.0)
    else:
        cod = pear
        dist = corr_distance(pear.to_numpy())

    # symmetrize defensively: squareform rejects tiny float asymmetry
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method=linkage_method)

    # Default k = number of real GICS sectors, so the A/B compares groupings of
    # equal granularity. Fewer/more clusters would make the dispersion
    # comparison meaningless (1 cluster trivially has huge dispersion; k=n has
    # none).
    sect = sector_labels(tickers)
    k_eff = int(k) if k else max(2, sect.nunique())
    labels = pd.Series(fcluster(link, t=k_eff, criterion="maxclust"),
                       index=tickers, name="cluster")
    order = quasi_diagonalize(link, len(tickers))
    seriation = pd.Series({tickers[p]: i for i, p in enumerate(order)},
                          name="hrp_order")

    # ── the measured bar: cluster dispersion vs sector dispersion ──
    common = [t for t in tickers if t in sect.index]
    base = within_group_dispersion(pear.loc[common, common], sect.loc[common])
    clus = within_group_dispersion(pear.loc[common, common], labels.loc[common])
    red = (1.0 - clus["dispersion"] / base["dispersion"]) * 100 if base["dispersion"] else float("nan")

    print(f"\n  universe compared: {len(common)} tickers, k={k_eff}")
    print(f"  sector  dispersion: {base['dispersion']:.4f} "
          f"({base['n_groups']} groups, mean within-corr {base['mean_within_corr']:.3f})")
    print(f"  cluster dispersion: {clus['dispersion']:.4f} "
          f"({clus['n_groups']} groups, mean within-corr {clus['mean_within_corr']:.3f})")
    print(f"  reduction: {red:+.1f}%  (bar >= 20%)  -> {'PASS' if red >= 20 else 'FAIL'}")

    out = pd.DataFrame({"ticker": tickers}).set_index("ticker")
    out["cluster"] = labels
    out["hrp_order"] = seriation
    out["sector"] = sect.reindex(tickers)
    out["metric"] = metric
    out["linkage"] = linkage_method
    out["k"] = k_eff
    out = out.reset_index()

    # Name each cluster by its dominant sector composition. These are the
    # "clustered sectors" that downstream consumers (peer_analytics, cross_section)
    # can use as a finer-grained alternative to GICS.
    names = _name_clusters(out)
    out["cluster_name"] = out["cluster"].map(names)

    disp = pd.DataFrame([
        {"grouping": "gics_sector", "metric": metric, **base},
        {"grouping": "hrp_cluster", "metric": metric, **clus},
        {"grouping": "reduction_pct", "metric": metric, "dispersion": red,
         "n_groups": k_eff, "n_pairs": clus["n_pairs"],
         "mean_within_corr": float("nan"), "singletons": 0},
    ])
    disp["bar_pct"] = 20.0
    disp["passes"] = bool(red >= 20)
    disp["n_assets"] = len(common)
    disp["years"] = years

    if save:
        out.to_parquet(OUT_CLUSTERS, index=False)
        disp.to_parquet(OUT_DISPERSION, index=False)
        print(f"\n  Saved {OUT_CLUSTERS} ({len(out)} rows)")
        print(f"  Saved {OUT_DISPERSION} ({len(disp)} rows)")

    return {"clusters": out, "dispersion": disp, "reduction_pct": red}


def sweep(metric: str = "corr", years_list=(3.0, 5.0),
          linkages=("average", "single", "complete", "ward"),
          max_assets: int | None = None, save: bool = False) -> pd.DataFrame:
    """Robustness sweep over linkage x lookback.

    Exists because the headline single-config number landed at exactly +20.0%,
    i.e. right on the bar. One cell on a bar is not a result: the sweep shows
    the reduction is linkage-dependent (single linkage fails badly), so the
    claim has to be reported as a range with the config attached.
    """
    import contextlib, io
    rows = []
    for lk in linkages:
        for yrs in years_list:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    r = run(metric=metric, years=yrs, linkage_method=lk,
                            max_assets=max_assets, save=False)
                rows.append({"metric": metric, "linkage": lk, "years": yrs,
                             "reduction_pct": r["reduction_pct"],
                             "passes": bool(r["reduction_pct"] >= 20)})
            except Exception as e:
                rows.append({"metric": metric, "linkage": lk, "years": yrs,
                             "reduction_pct": float("nan"), "passes": False,
                             "error": str(e)[:120]})
    df = pd.DataFrame(rows)
    print(f"\n  Robustness sweep ({metric}):")
    for _, r in df.iterrows():
        print(f"    {r['linkage']:9s} {r['years']:.1f}y  {r['reduction_pct']:+6.1f}%  "
              f"{'PASS' if r['passes'] else 'FAIL'}")
    ok = df["passes"].sum()
    print(f"    -> {ok}/{len(df)} configs clear the 20% bar "
          f"(range {df['reduction_pct'].min():+.1f}% to {df['reduction_pct'].max():+.1f}%)")
    if save:
        df.to_parquet(OUT_SWEEP, index=False)
        print(f"  Saved {OUT_SWEEP} ({len(df)} rows)")
    return df


def main():
    ap = argparse.ArgumentParser(description="HRP / codependence regime clustering")
    ap.add_argument("--metric", choices=["corr", "dcor"], default="corr",
                    help="corr = Pearson distance; dcor = distance correlation (non-linear)")
    ap.add_argument("--k", type=int, default=None,
                    help="number of clusters (default: number of GICS sectors, for a fair A/B)")
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--min-cov", type=float, default=0.95)
    ap.add_argument("--linkage", dest="linkage_method", default="average",
                    choices=["single", "average", "complete", "ward"])
    ap.add_argument("--max-assets", type=int, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="run the linkage x lookback robustness sweep instead of one config")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    if args.sweep:
        sweep(metric=args.metric, max_assets=args.max_assets, save=args.save)
        return 0
    run(metric=args.metric, k=args.k, years=args.years, min_cov=args.min_cov,
        linkage_method=args.linkage_method, max_assets=args.max_assets, save=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
