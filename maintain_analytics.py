#!/usr/bin/env python3
"""
maintain_analytics.py — Regenerate all analysis CSV files from parquet sources.

CSV outputs maintained by this program:
  sector_correlation_matrix.csv
  fertilizer_correlation_matrix.csv
  rolling_sector_correlations.csv
  correlation_stability_metrics.csv
  hmm_2state_regimes.csv
  hmm_2state_regime_correlations.csv
  kalman_correlations.csv
  granger_causality_sectors.csv
  index_backtest_stats.csv
  index_levels_1y.parquet  (levels used by backtest)
  vol_target_vs_risk_parity.csv
  growth_ai_vol_vs_risk_parity.csv
  erc_gmv_strategies.csv
  erc_gmv_summary.csv

Usage:
  python maintain_analytics.py all
  python maintain_analytics.py correlations
  python maintain_analytics.py rolling
  python maintain_analytics.py stability
  python maintain_analytics.py hmm
  python maintain_analytics.py kalman
  python maintain_analytics.py var
  python maintain_analytics.py backtest
  python maintain_analytics.py list
  python maintain_analytics.py vol-rp
  python maintain_analytics.py optimize
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent

PRICES_FILE = DATA_DIR / "daily_prices/"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
HOLDINGS_FILE = DATA_DIR / "portfolio_holdings.parquet"
INDEX_LEVELS_FILE = DATA_DIR / "index_levels_1y.parquet"

# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------

def load_prices() -> pd.DataFrame:
    if not PRICES_FILE.exists():
        raise SystemExit(f"Missing {PRICES_FILE}")
    df = pd.read_parquet(PRICES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_stocks() -> pd.DataFrame:
    if not STOCKS_FILE.exists():
        raise SystemExit(f"Missing {STOCKS_FILE}")
    return pd.read_parquet(STOCKS_FILE)


def load_holdings() -> pd.DataFrame:
    if HOLDINGS_FILE.exists():
        return pd.read_parquet(HOLDINGS_FILE)
    return pd.DataFrame(columns=["ticker"])


def wide_closes(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pivot_table(index="date", columns="ticker", values="close").sort_index()


def sector_returns(prices: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    wide = wide_closes(prices)
    logrets = np.log(wide / wide.shift(1))
    out = {}
    for sector, grp in stocks.groupby("sector"):
        cols = [t for t in grp["ticker"] if t in logrets.columns]
        if cols:
            out[sector] = logrets[cols].mean(axis=1)
    sec = pd.DataFrame(out).dropna(how="all")
    return sec.dropna(thresh=max(3, len(sec.columns) // 2))


ABBR = {
    "Communication Services": "CommSvc",
    "Consumer Discretionary": "ConsDisc",
    "Consumer Staples": "ConsStpl",
    "Energy": "Energy",
    "Financials": "Fins",
    "Health Care": "HlthCare",
    "Industrials": "Industrials",
    "Information Technology": "InfoTech",
    "Materials": "Materials",
    "Real Estate": "REITs",
    "Utilities": "Utilities",
}

KEY_PAIRS = [
    ("Materials", "Consumer Staples"),
    ("Materials", "Health Care"),
    ("Materials", "Energy"),
    ("Materials", "Financials"),
    ("Consumer Staples", "Health Care"),
    ("Consumer Staples", "Utilities"),
    ("Health Care", "Utilities"),
    ("Industrials", "Utilities"),
    ("Information Technology", "Real Estate"),
    ("Consumer Discretionary", "Information Technology"),
    ("Energy", "Utilities"),
    ("Financials", "Real Estate"),
]


def cmd_correlations(_args=None):
    prices = load_prices()
    stocks = load_stocks()
    sec = sector_returns(prices, stocks)
    corr = sec.corr()
    corr.to_parquet(DATA_DIR / "sector_correlation_matrix.parquet")
    print(f"Wrote sector_correlation_matrix.csv  ({corr.shape[0]}x{corr.shape[1]})")

    fert = stocks[stocks.get("index_member", False) == True]["ticker"].tolist()
    wide = wide_closes(prices)
    logrets = np.log(wide / wide.shift(1))
    fert = [t for t in fert if t in logrets.columns]
    if len(fert) >= 2:
        fcorr = logrets[fert].corr()
        fcorr.to_parquet(DATA_DIR / "fertilizer_correlation_matrix.parquet")
        print(f"Wrote fertilizer_correlation_matrix.csv  ({len(fert)} members)")
    else:
        print("Skipped fertilizer matrix (need >=2 index members with prices)")


def cmd_rolling(_args=None):
    prices = load_prices()
    stocks = load_stocks()
    sec = sector_returns(prices, stocks)

    out = {}
    for a, b in KEY_PAIRS:
        if a not in sec.columns or b not in sec.columns:
            continue
        label = f"{ABBR.get(a, a)}_{ABBR.get(b, b)}_r10"
        out[label] = sec[a].rolling(10, min_periods=8).corr(sec[b])
        label20 = f"{ABBR.get(a, a)}_{ABBR.get(b, b)}_r20"
        out[label20] = sec[a].rolling(20, min_periods=15).corr(sec[b])

    df = pd.DataFrame(out)
    df.to_parquet(DATA_DIR / "rolling_sector_correlations.parquet")
    print(f"Wrote rolling_sector_correlations.csv  ({df.shape[1]} series, {len(df)} days)")


def cmd_stability(_args=None):
    prices = load_prices()
    stocks = load_stocks()
    sec = sector_returns(prices, stocks)

    rows = []
    for a, b in KEY_PAIRS:
        if a not in sec.columns or b not in sec.columns:
            continue
        r = sec[a].rolling(10, min_periods=8).corr(sec[b]).dropna()
        if len(r) < 5:
            continue
        mean, std = r.mean(), r.std()
        rows.append({
            "pair": f"{ABBR.get(a, a)} x {ABBR.get(b, b)}",
            "sector_a": a,
            "sector_b": b,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "range": round(r.max() - r.min(), 4),
            "iqr": round(r.quantile(0.75) - r.quantile(0.25), 4),
            "frac_pos": round((r > 0).mean(), 4),
            "abs_mean": round(r.abs().mean(), 4),
            "cv": round(abs(std / mean), 4) if abs(mean) > 0.05 else np.nan,
            "stability": round(max(0.0, 1.0 - std), 4),
            "n_windows": len(r),
        })

    df = pd.DataFrame(rows).sort_values("std", ascending=False)
    df.to_parquet(DATA_DIR / "correlation_stability_metrics.parquet")
    print(f"Wrote correlation_stability_metrics.csv  ({len(df)} pairs)")


def cmd_hmm(_args=None):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("hmmlearn not installed — pip install hmmlearn")
        return

    prices = load_prices()
    stocks = load_stocks()
    sec = sector_returns(prices, stocks)

    mkt = sec.mean(axis=1).dropna()
    feat = pd.DataFrame({
        "mkt": mkt,
        "abs_mkt": mkt.abs(),
        "materials": sec["Materials"].reindex(mkt.index) if "Materials" in sec.columns else np.nan,
        "staples": sec["Consumer Staples"].reindex(mkt.index) if "Consumer Staples" in sec.columns else np.nan,
    }).dropna()

    X = feat[["mkt", "abs_mkt", "materials", "staples"]].values
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)

    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=300, random_state=7)
    model.fit(Xz)
    states = model.predict(Xz)
    feat = feat.copy()
    feat["state"] = states

    stats = feat.groupby("state").agg(mean_abs=("abs_mkt", "mean"))
    calm_id = int(stats["mean_abs"].idxmin())
    stress_id = int(stats["mean_abs"].idxmax())
    labels = {calm_id: "Calm", stress_id: "Stress"}
    feat["regime"] = feat["state"].map(labels)

    feat[["regime", "state", "mkt", "abs_mkt"]].to_parquet(DATA_DIR / "hmm_2state_regimes.parquet")
    print(f"Wrote hmm_2state_regimes.csv  (Calm={(feat.regime=='Calm').sum()}, Stress={(feat.regime=='Stress').sum()})")

    sec_a = sec.reindex(feat.index)
    rows = []
    for a, b in KEY_PAIRS:
        if a not in sec_a.columns or b not in sec_a.columns:
            continue
        c_calm = sec_a.loc[feat.regime == "Calm", a].corr(sec_a.loc[feat.regime == "Calm", b])
        c_stress = sec_a.loc[feat.regime == "Stress", a].corr(sec_a.loc[feat.regime == "Stress", b])
        c_all = sec_a[a].corr(sec_a[b])
        rows.append({
            "pair": f"{ABBR.get(a, a)} x {ABBR.get(b, b)}",
            "corr_calm": round(c_calm, 4) if pd.notna(c_calm) else np.nan,
            "corr_stress": round(c_stress, 4) if pd.notna(c_stress) else np.nan,
            "corr_all": round(c_all, 4) if pd.notna(c_all) else np.nan,
            "delta_stress_calm": round(c_stress - c_calm, 4) if pd.notna(c_stress) and pd.notna(c_calm) else np.nan,
        })
    pd.DataFrame(rows).to_parquet(DATA_DIR / "hmm_2state_regime_correlations.parquet")
    print(f"Wrote hmm_2state_regime_correlations.csv  ({len(rows)} pairs)")


def kalman_corr(x, y, q=1e-4, r=0.05):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    rho_hat = np.full(n, np.nan)
    P, rho = 1.0, 0.0
    for t in range(n):
        if t < 5:
            continue
        sx = np.std(x[max(0, t - 19): t + 1]) + 1e-12
        sy = np.std(y[max(0, t - 19): t + 1]) + 1e-12
        z = np.clip((x[t] * y[t]) / (sx * sy), -3, 3)
        P = P + q
        K = P / (P + r)
        rho = rho + K * (z - rho)
        P = (1 - K) * P
        rho_hat[t] = np.clip(rho, -1, 1)
    return rho_hat


def cmd_kalman(_args=None):
    prices = load_prices()
    stocks = load_stocks()
    sec = sector_returns(prices, stocks)

    pairs = [
        ("Materials", "Consumer Staples", "MatxStaples"),
        ("Materials", "Health Care", "MatxHealth"),
        ("Consumer Staples", "Health Care", "StaplesxHealth"),
        ("Industrials", "Utilities", "IndxUtils"),
        ("Materials", "Financials", "MatxFins"),
        ("Energy", "Utilities", "EnergyxUtils"),
    ]
    out = {}
    for a, b, label in pairs:
        if a not in sec.columns or b not in sec.columns:
            continue
        al = pd.concat([sec[a], sec[b]], axis=1).dropna()
        out[label] = pd.Series(
            kalman_corr(al.iloc[:, 0].values, al.iloc[:, 1].values),
            index=al.index,
        )

    if INDEX_LEVELS_FILE.exists():
        try:
            idx = pd.read_parquet(INDEX_LEVELS_FILE)
            idx["date"] = pd.to_datetime(idx["date"])
            idx = idx.set_index("date")
            idx_rets = idx.pct_change().dropna()
            for a, b in [("Fertilizer", "Defensive"), ("Fertilizer", "Personal"), ("Defensive", "Personal")]:
                if a in idx_rets.columns and b in idx_rets.columns:
                    al = pd.concat([idx_rets[a], idx_rets[b]], axis=1).dropna()
                    out[f"{a}x{b}"] = pd.Series(
                        kalman_corr(al.iloc[:, 0].values, al.iloc[:, 1].values, q=5e-5, r=0.08),
                        index=al.index,
                    )
        except Exception as e:
            print(f"  (index Kalman skipped: {e})")

    df = pd.DataFrame(out)
    df.to_parquet(DATA_DIR / "kalman_correlations.parquet")
    print(f"Wrote kalman_correlations.csv  ({df.shape[1]} series, {len(df)} days)")


def cmd_var(_args=None):
    try:
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        print("statsmodels not installed — pip install statsmodels")
        return

    prices = load_prices()
    stocks = load_stocks()
    sec = sector_returns(prices, stocks)

    var_secs = [s for s in [
        "Materials", "Consumer Staples", "Health Care", "Energy",
        "Financials", "Utilities", "Industrials",
    ] if s in sec.columns]
    var_data = sec[var_secs].dropna()
    if len(var_data) < 30:
        print("Not enough history for VAR")
        return

    model = VAR(var_data)
    lag_sel = model.select_order(maxlags=min(8, max(2, len(var_data) // 10)))
    best_lag = lag_sel.aic if (lag_sel.aic and lag_sel.aic > 0) else 2
    best_lag = int(min(max(best_lag, 1), 5))

    gc = pd.DataFrame(index=var_secs, columns=var_secs, dtype=float)
    for caused in var_secs:
        for causing in var_secs:
            if causing == caused:
                continue
            try:
                gt = grangercausalitytests(
                    var_data[[caused, causing]].dropna(),
                    maxlag=best_lag,
                    verbose=False,
                )
                gc.loc[causing, caused] = min(
                    gt[i + 1][0]["ssr_ftest"][1] for i in range(best_lag)
                )
            except Exception:
                pass

    gc.to_parquet(DATA_DIR / "granger_causality_sectors.parquet")
    print(f"Wrote granger_causality_sectors.csv  (VAR lag={best_lag}, {len(var_secs)} sectors)")


def cmd_backtest(_args=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    prices = load_prices()
    stocks = load_stocks()
    holdings = load_holdings()
    wide = wide_closes(prices)
    rets = wide.pct_change()

    def ew_index(tickers, name):
        cols = [t for t in tickers if t in rets.columns]
        if not cols:
            return pd.Series(dtype=float, name=name)
        r = rets[cols].mean(axis=1)
        idx = (1 + r.fillna(0)).cumprod() * 100
        first = idx.first_valid_index()
        if first is not None:
            idx = idx / idx.loc[first] * 100
        idx.name = name
        return idx

    fert = stocks[stocks.get("index_member", False) == True]["ticker"].tolist()
    defn = (
        stocks[stocks.get("defensive_value_index", False) == True]["ticker"].tolist()
        if "defensive_value_index" in stocks.columns else []
    )
    port = holdings["ticker"].tolist() if "ticker" in holdings.columns else []

    idx = pd.concat([
        ew_index(fert, "Fertilizer"),
        ew_index(defn, "Defensive"),
        ew_index(port, "Personal"),
    ], axis=1).dropna(how="any")

    out = idx.reset_index().rename(columns={"index": "date"})
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), INDEX_LEVELS_FILE)
    print(f"Wrote index_levels_1y.parquet  ({len(idx)} days)")

    def perf_stats(series, label, rf=0.04):
        s = series.dropna()
        if len(s) < 20:
            return None
        r = s.pct_change().dropna()
        total = s.iloc[-1] / s.iloc[0] - 1
        n_years = max((s.index[-1] - s.index[0]).days / 365.25, 0.01)
        ann_ret = (1 + total) ** (1 / n_years) - 1
        ann_vol = r.std() * np.sqrt(252)
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
        max_dd = (s / s.cummax() - 1).min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
        return {
            "index": label,
            "start": s.index[0].date().isoformat(),
            "end": s.index[-1].date().isoformat(),
            "total_return_pct": round(total * 100, 2),
            "ann_return_pct": round(ann_ret * 100, 2),
            "ann_vol_pct": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_dd_pct": round(max_dd * 100, 2),
            "calmar": round(calmar, 2),
            "final_level": round(s.iloc[-1], 2),
            "n_members": {"Fertilizer": len(fert), "Defensive": len(defn), "Personal": len(port)}.get(label),
        }

    stats = [perf_stats(idx[c], c) for c in idx.columns]
    stats = [s for s in stats if s]
    pd.DataFrame(stats).to_parquet(DATA_DIR / "index_backtest_stats.parquet")
    print(f"Wrote index_backtest_stats.csv  ({len(stats)} indices)")
    for s in stats:
        print(f"  {s['index']}: total {s['total_return_pct']:+.1f}%  Sharpe {s['sharpe']:.2f}  MaxDD {s['max_dd_pct']:.1f}%")




def cmd_growth_tech(_args=None):
    """Full growth/tech index analysis suite."""
    import subprocess, sys
    script = DATA_DIR / "growth_tech_analytics.py"
    r = subprocess.run([sys.executable, str(script)], cwd=str(DATA_DIR))
    if r.returncode != 0:
        raise SystemExit("growth_tech_analytics failed")

def cmd_optimize(_args=None):

    """ERC risk parity + GMV minimum variance strategies."""
    import subprocess, sys
    script = DATA_DIR / "portfolio_optimization.py"
    r = subprocess.run([sys.executable, str(script), "--universe", "portfolio"], cwd=str(DATA_DIR))
    if r.returncode != 0:
        raise SystemExit("portfolio_optimization failed")

def cmd_vol_rp(_args=None):

    """Vol targeting vs risk parity comparison CSVs."""
    import subprocess, sys
    script = DATA_DIR / "risk_parity_analytics.py"
    r = subprocess.run([sys.executable, str(script)], cwd=str(DATA_DIR))
    if r.returncode != 0:
        raise SystemExit("risk_parity_analytics failed")

def cmd_cross_asset(_args=None):
    """Delegate to cross_asset_analysis.py all."""
    import subprocess, sys
    r = subprocess.run([sys.executable, str(DATA_DIR / "cross_asset_analysis.py"), "all"], cwd=str(DATA_DIR))
    if r.returncode != 0:
        print("cross_asset_analysis failed")

def cmd_all(_args=None):
    print("=== Regenerating all analysis CSVs ===\n")
    cmd_correlations()
    cmd_rolling()
    cmd_stability()
    cmd_hmm()
    cmd_backtest()
    cmd_kalman()
    cmd_var()
    cmd_cross_asset()
    cmd_vol_rp()
    cmd_optimize()
    cmd_growth_tech()
    print("\n=== Done ===")
    cmd_list()


def cmd_list(_args=None):
    csvs = sorted(DATA_DIR.glob("*.csv"))
    print("\nCSV files in stock_monitor/:")
    for p in csvs:
        sz = p.stat().st_size
        print(f"  {p.name:<45} {sz:>8} bytes")
    if INDEX_LEVELS_FILE.exists():
        print(f"  {INDEX_LEVELS_FILE.name:<45} {INDEX_LEVELS_FILE.stat().st_size:>8} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Maintain analysis CSV files from parquet sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("all", help="Regenerate every analysis CSV").set_defaults(func=cmd_all)
    sub.add_parser("correlations", help="Sector + fertilizer correlation matrices").set_defaults(func=cmd_correlations)
    sub.add_parser("rolling", help="Rolling 10d/20d sector correlations").set_defaults(func=cmd_rolling)
    sub.add_parser("stability", help="Rolling correlation stability metrics").set_defaults(func=cmd_stability)
    sub.add_parser("hmm", help="2-state HMM regimes + conditional corrs").set_defaults(func=cmd_hmm)
    sub.add_parser("kalman", help="Kalman filter dynamic correlations").set_defaults(func=cmd_kalman)
    sub.add_parser("var", help="VAR lag selection + Granger causality matrix").set_defaults(func=cmd_var)
    sub.add_parser("backtest", help="Rebuild index levels + performance stats").set_defaults(func=cmd_backtest)
    sub.add_parser("list", help="List maintained CSV files").set_defaults(func=cmd_list)
    sub.add_parser("cross-asset", help="Cross-asset/sector correlations + sector prices").set_defaults(func=cmd_cross_asset)
    sub.add_parser("vol-rp", help="Vol targeting vs risk parity CSVs").set_defaults(func=cmd_vol_rp)
    sub.add_parser("optimize", help="ERC risk parity + GMV minimum variance").set_defaults(func=cmd_optimize)
    sub.add_parser("growth-tech", help="Growth/tech index full analysis suite").set_defaults(func=cmd_growth_tech)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
