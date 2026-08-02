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


def main():
    parser = argparse.ArgumentParser(
        description="Kelly criterion position-sizing estimators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("continuous", help="f* = (μ − r) / σ²")
    p.add_argument("--mu", type=float, help="Expected annualized return (decimal, e.g. 0.13)")
    p.add_argument("--sigma", type=float, help="Annualized volatility (decimal, e.g. 0.35)")
    p.add_argument("--r", type=float, default=0.04, help="Risk-free rate (decimal, default 0.04)")
    p.add_argument("--ticker", help="Use stored mid-point params for this ticker")
    p.add_argument("--fraction", type=float, help="Also show custom fraction of full Kelly")
    p.set_defaults(func=cmd_continuous)

    p = sub.add_parser("binary", help="f* = (b·p − q) / b")
    p.add_argument("--p", type=float, required=True, help="Win probability (0–1)")
    p.add_argument("--b", type=float, required=True, help="Net odds (profit per unit risked)")
    p.set_defaults(func=cmd_binary)

    p = sub.add_parser("show", help="Show stored Kelly parameters")
    p.add_argument("--ticker")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("pypl", help="Reproduce the attached PayPal example")
    p.set_defaults(func=cmd_pypl_example)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
