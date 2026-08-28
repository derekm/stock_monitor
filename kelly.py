#!/usr/bin/env python3
"""
kelly.py - Kelly criterion estimators for position sizing.

Supports:
  1. Continuous (geometric Brownian) form used for stocks:
       f* = (μ - r) / σ²
     where μ, r, σ are annualized decimals.

  2. Binary / edge form (gambling-style):
       f* = (b·p - q) / b
     where p = win prob, q = 1-p, b = net odds (profit per unit risked).

  3. Fractional Kelly (½, ¼, or custom fraction) for practical risk control.

  4. Lookup / store of parameters in kelly_parameters.parquet.

Examples:
  python kelly.py continuous --mu 0.13 --sigma 0.35 --r 0.04
  python kelly.py continuous --ticker PYPL
  python kelly.py binary --p 0.60 --b 1.67
  python kelly.py show
  python kelly.py show --ticker PYPL
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
KELLY_FILE = DATA_DIR / "kelly_parameters.parquet"


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def continuous_kelly(mu: float, sigma: float, r: float = 0.0) -> dict:
    """
    f* = (μ - r) / σ²

    Parameters are annualized decimals (e.g. mu=0.13 for 13%).
    Returns full Kelly and common fractional sizes.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    edge = mu - r
    f_star = edge / (sigma ** 2)
    # Cap display of extreme values; caller can still use raw
    return {
        "mu": mu,
        "sigma": sigma,
        "r": r,
        "edge": edge,
        "f_star": f_star,
        "half_kelly": f_star * 0.5,
        "quarter_kelly": f_star * 0.25,
        "growth_approx_full": edge * f_star - 0.5 * (sigma * f_star) ** 2,  # expected log growth
    }


def binary_kelly(p: float, b: float) -> dict:
    """
    f* = (b·p - q) / b
    p = probability of winning, b = net fractional odds (profit / amount risked).
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if b <= 0:
        raise ValueError("b (net odds) must be positive")
    q = 1.0 - p
    f_star = (b * p - q) / b
    return {
        "p": p,
        "q": q,
        "b": b,
        "f_star": f_star,
        "half_kelly": f_star * 0.5,
        "quarter_kelly": f_star * 0.25,
    }


def fractional(f_star: float, fraction: float = 0.5) -> float:
    """Scale full Kelly by a fraction (0.5 = half Kelly, etc.)."""
    return f_star * fraction


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_kelly_table() -> pd.DataFrame:
    if KELLY_FILE.exists():
        return pd.read_parquet(KELLY_FILE)
    return pd.DataFrame()


def params_for_ticker(ticker: str) -> dict | None:
    """Return mid-point μ, σ, r for a ticker from kelly_parameters.parquet."""
    df = load_kelly_table()
    if df.empty or ticker.upper() not in df["ticker"].values:
        return None
    row = df[df["ticker"] == ticker.upper()].iloc[0]
    mu = (float(row["mu_low_pct"]) + float(row["mu_high_pct"])) / 200.0  # avg of % → decimal
    sigma = (float(row["sigma_low_pct"]) + float(row["sigma_high_pct"])) / 200.0
    r = float(row["r_pct"]) / 100.0 if "r_pct" in row and pd.notna(row.get("r_pct")) else 0.04
    return {"mu": mu, "sigma": sigma, "r": r, "row": row}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_continuous(args):
    if args.ticker:
        params = params_for_ticker(args.ticker)
        if not params:
            print(f"No Kelly parameters stored for {args.ticker}. Use --mu/--sigma or add via table.")
            return
        mu, sigma, r = params["mu"], params["sigma"], params["r"]
        print(f"Using stored mid-point params for {args.ticker.upper()}:")
        print(f"  μ ≈ {mu*100:.1f}%  σ ≈ {sigma*100:.1f}%  r ≈ {r*100:.1f}%")
    else:
        mu = args.mu
        sigma = args.sigma
        r = args.r
        if mu is None or sigma is None:
            print("Provide --mu and --sigma, or --ticker")
            return

    res = continuous_kelly(mu, sigma, r)
    print("\nContinuous Kelly (stocks / GBM approximation)")
    print(f"  Edge (μ − r)     : {res['edge']*100:.2f}%")
    print(f"  Full Kelly f*    : {res['f_star']*100:.1f}% of capital")
    print(f"  Half Kelly       : {res['half_kelly']*100:.1f}%")
    print(f"  Quarter Kelly    : {res['quarter_kelly']*100:.1f}%")
    if args.fraction is not None:
        print(f"  Custom ({args.fraction:.2f}×) : {fractional(res['f_star'], args.fraction)*100:.1f}%")
    print(f"  Approx E[log growth] at full Kelly: {res['growth_approx_full']*100:.2f}%")
    print("\nNote: Full Kelly is aggressive. Prefer ½ or ¼ Kelly in practice.")
    print("      Overestimating μ or underestimating σ produces oversized bets.")


def cmd_binary(args):
    res = binary_kelly(args.p, args.b)
    print("\nBinary / Edge Kelly")
    print(f"  p (win prob)     : {res['p']*100:.1f}%")
    print(f"  b (net odds)     : {res['b']:.3f}")
    print(f"  Full Kelly f*    : {res['f_star']*100:.1f}%")
    print(f"  Half Kelly       : {res['half_kelly']*100:.1f}%")
    print(f"  Quarter Kelly    : {res['quarter_kelly']*100:.1f}%")


def cmd_show(args):
    df = load_kelly_table()
    if df.empty:
        print("No kelly_parameters.parquet yet.")
        return
    if args.ticker:
        df = df[df["ticker"] == args.ticker.upper()]
    cols = [c for c in [
        "ticker", "mu_low_pct", "mu_high_pct", "sigma_low_pct", "sigma_high_pct",
        "r_pct", "f_star_approx", "half_kelly_low_pct", "half_kelly_high_pct",
        "style", "source"
    ] if c in df.columns]
    print(df[cols].to_string(index=False))


def cmd_pypl_example(args):
    """Reproduce the attached PayPal worked example."""
    print("PayPal (PYPL) — worked example from attached analysis")
    print("Assumptions: μ ≈ 13% (mid of 12–15%), σ ≈ 35%, r ≈ 4%")
    res = continuous_kelly(0.13, 0.35, 0.04)
    print(f"\n  f* = (0.13 - 0.04) / (0.35)² = 0.09 / 0.1225 ≈ {res['f_star']:.2f} ({res['f_star']*100:.0f}%)")
    print(f"  Half Kelly   ≈ {res['half_kelly']*100:.0f}%")
    print(f"  Quarter Kelly≈ {res['quarter_kelly']*100:.0f}%")
    print("\nPractical recommendation in source text: 10–25% portfolio allocation")
    print("(fractional Kelly, reflecting uncertainty in edge and elevated fintech vol).")
    print("Current price context ~$46; low P/B ~2x; strong FCF supporting buybacks.")


def cmd_leverage_space(args):
    tmi = pd.read_parquet(DATA_DIR / "bogle_tmi.parquet")
    bpi = pd.read_parquet(DATA_DIR / "bogle_bpi.parquet")
    tmi["date"] = pd.to_datetime(tmi["date"])
    bpi["date"] = pd.to_datetime(bpi["date"])
    m = tmi.merge(bpi, on="date", suffixes=("_t", "_b"))
    rt = m["ret_net_t"].fillna(0).to_numpy()
    rb = m["ret_net_b"].fillna(0).to_numpy()
    grid = np.linspace(0.0, 1.5, 16)
    rows = []
    best = (-np.inf, 0, 0)
    for ft in grid:
        for fb in grid:
            if ft + fb > 1.5:
                continue
            wealth = np.prod(1 + ft * rt + fb * rb)
            if not np.isfinite(wealth) or wealth <= 0:
                continue
            rows.append({"f_tmi": ft, "f_bpi": fb, "terminal": float(wealth)})
            if wealth > best[0]:
                best = (wealth, ft, fb)
    out = pd.DataFrame(rows)
    out.to_parquet(DATA_DIR / "leverage_space_allocation.parquet", index=False)
    print(f"max terminal {best[0]:.2f} at f_tmi={best[1]:.2f} f_bpi={best[2]:.2f}  n={len(out)}")


def cmd_multi_period(args):
    tmi = pd.read_parquet(DATA_DIR / "bogle_tmi.parquet")
    r = pd.to_numeric(tmi["ret_net"], errors="coerce").dropna().to_numpy()
    mu, sig = float(r.mean() * 252), float(r.std() * np.sqrt(252))
    f_single = max(0.0, (mu - 0.04) / (sig ** 2)) if sig > 0 else 0.0
    # multi-period: f* ≈ (μ − σ²/2 − r) / σ²  (vol-drag)
    f_mp = max(0.0, (mu - 0.5 * sig ** 2 - 0.04) / (sig ** 2)) if sig > 0 else 0.0
    out = pd.DataFrame([
        {"kind": "single", "f": f_single, "mu": mu, "sig": sig},
        {"kind": "multi_period", "f": f_mp, "mu": mu, "sig": sig},
    ])
    out.to_parquet(DATA_DIR / "multi_period_kelly.parquet", index=False)
    print(out.to_string(index=False))


def cmd_ls_vs_erc(args):
    """Vince 2-asset grid vs equal-risk-contribution on TMI/BPI. Block-bootstrap."""
    tmi = pd.read_parquet(DATA_DIR / "bogle_tmi.parquet")
    bpi = pd.read_parquet(DATA_DIR / "bogle_bpi.parquet")
    t = pd.to_datetime(tmi["date"])
    b = pd.to_datetime(bpi["date"])
    a = tmi.assign(date=t)[["date", "ret_net"]].rename(columns={"ret_net": "tmi"})
    c = bpi.assign(date=b)[["date", "ret_net"]].rename(columns={"ret_net": "bpi"})
    m = a.merge(c, on="date").dropna()
    rt = pd.to_numeric(m["tmi"], errors="coerce").to_numpy()
    rb = pd.to_numeric(m["bpi"], errors="coerce").to_numpy()
    ok = np.isfinite(rt) & np.isfinite(rb)
    rt, rb = rt[ok], rb[ok]
    # ERC: w ∝ 1/σ
    st, sb = float(rt.std()), float(rb.std())
    w_t = (1 / st) / (1 / st + 1 / sb) if st > 0 and sb > 0 else 0.5
    w_b = 1.0 - w_t
    # Vince grid already on disk if present; else 1.50 / 0
    ls_path = DATA_DIR / "leverage_space_allocation.parquet"
    if ls_path.exists():
        g = pd.read_parquet(ls_path)
        best = g.loc[g["terminal"].idxmax()]
        ft, fb = float(best["f_tmi"]), float(best["f_bpi"])
    else:
        ft, fb = 1.50, 0.0
    rng = np.random.default_rng(0)
    n, block, paths = len(rt), 21, 400
    nblk = n // block
    def path_term(w1, w2):
        terms = []
        for _ in range(paths):
            idx = rng.integers(0, nblk, size=nblk)
            r1 = np.concatenate([rt[i * block:(i + 1) * block] for i in idx])
            r2 = np.concatenate([rb[i * block:(i + 1) * block] for i in idx])
            wealth = np.prod(1 + w1 * r1 + w2 * r2)
            terms.append(wealth if np.isfinite(wealth) and wealth > 0 else np.nan)
        return np.asarray(terms)

    ls = path_term(ft, fb)
    erc = path_term(w_t, w_b)
    def stats(x, name):
        x = x[np.isfinite(x)]
        return {"book": name, "median": float(np.median(x)), "p05": float(np.quantile(x, 0.05)),
                "mean": float(np.mean(x)), "n_paths": int(len(x))}
    out = pd.DataFrame([stats(ls, f"ls_{ft:.2f}_{fb:.2f}"), stats(erc, f"erc_{w_t:.2f}_{w_b:.2f}")])
    out.to_parquet(DATA_DIR / "ls_vs_erc.parquet", index=False)
    print(out.to_string(index=False))
    print(f"LS median {out.iloc[0]['median']:.3f} vs ERC {out.iloc[1]['median']:.3f}")


def cmd_leverage_space_multi(args):
    """Vince multi-asset Leverage Space: joint optimal f across TMI/BPI/QMI.
    
    Unlike the 2-asset grid (marginal Kelly per asset), this implements Vince's
    full Leverage Space: grid search over the joint leverage space with a total
    leverage cap, maximizing median terminal wealth across block-bootstrap paths.
    The joint distribution preserves cross-asset codependence.
    """
    funds = {
        "tmi": pd.read_parquet(DATA_DIR / "bogle_tmi.parquet"),
        "bpi": pd.read_parquet(DATA_DIR / "bogle_bpi.parquet"),
        "qmi": pd.read_parquet(DATA_DIR / "bogle_qmi.parquet"),
    }
    for k in funds:
        funds[k]["date"] = pd.to_datetime(funds[k]["date"])
    
    # Align on common dates
    merged = None
    for k, df in funds.items():
        d = df[["date", "ret_net"]].rename(columns={"ret_net": k})
        merged = d if merged is None else merged.merge(d, on="date")
    merged = merged.dropna().sort_values("date")
    print(f"Common history: {merged['date'].min()} → {merged['date'].max()} ({len(merged)} days)")
    
    rets = {k: merged[k].to_numpy(dtype=float) for k in funds}
    n = len(merged)
    
    # Grid: each f in [0, 1.5], total <= 2.0
    grid = np.linspace(0.0, 1.5, 7)
    cap = 2.0
    paths = 500
    block = 21
    rng = np.random.default_rng(42)
    nblk = n // block
    
    best_median = -np.inf
    best_fs = None
    all_rows = []
    
    # 3D grid search — 7^3 = 343 combinations
    for ft in grid:
        for fb in grid:
            for fq in grid:
                if ft + fb + fq > cap:
                    continue
                fs = {"tmi": ft, "bpi": fb, "qmi": fq}
                
                # Vectorized block-bootstrap: all paths at once
                idx = rng.integers(0, nblk, size=(paths, nblk))
                path_rets = np.zeros((paths, nblk * block))
                for k in funds:
                    fund_blocks = rets[k][:nblk * block].reshape(nblk, block)
                    sampled = fund_blocks[idx.ravel()].reshape(paths, nblk * block)
                    path_rets += fs[k] * sampled
                
                wealth = np.prod(1 + path_rets, axis=1)
                valid = wealth[np.isfinite(wealth) & (wealth > 0)]
                if len(valid) < paths * 0.5:
                    continue
                med = float(np.median(valid))
                all_rows.append({**fs, "median_terminal": med, "mean_terminal": float(np.mean(valid)),
                                 "n_paths": int(len(valid))})
                if med > best_median:
                    best_median = med
                    best_fs = fs
    
    out = pd.DataFrame(all_rows).sort_values("median_terminal", ascending=False)
    out.to_parquet(DATA_DIR / "leverage_space_sizing.parquet", index=False)
    print(f"\nBest: f_tmi={best_fs['tmi']:.2f}, f_bpi={best_fs['bpi']:.2f}, f_qmi={best_fs['qmi']:.2f}")
    print(f"Median terminal: {best_median:.2f}")
    print(f"Wrote leverage_space_sizing.parquet ({len(out)} combos)")
    print(f"\nTop 10:")
    print(out.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Kelly criterion position-sizing estimators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("continuous", help="f* = (μ − r) / σ²")
    p.add_argument("--mu", type=float)
    p.add_argument("--sigma", type=float)
    p.add_argument("--r", type=float, default=0.04)
    p.add_argument("--ticker")
    p.add_argument("--fraction", type=float)
    p.set_defaults(func=cmd_continuous)

    p = sub.add_parser("binary")
    p.add_argument("--p", type=float, required=True)
    p.add_argument("--b", type=float, required=True)
    p.set_defaults(func=cmd_binary)

    p = sub.add_parser("show")
    p.add_argument("--ticker")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("pypl")
    p.set_defaults(func=cmd_pypl_example)

    p = sub.add_parser("leverage-space")
    p.set_defaults(func=cmd_leverage_space)

    p = sub.add_parser("multi-period")
    p.set_defaults(func=cmd_multi_period)

    p = sub.add_parser("ls-vs-erc")
    p.set_defaults(func=cmd_ls_vs_erc)

    p = sub.add_parser("leverage-space-multi", help="Multi-asset Leverage Space (3-asset Vince grid)")
    p.set_defaults(func=cmd_leverage_space_multi)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
