#!/usr/bin/env python3
"""
portfolio_optimization.py — ERC risk parity & Global Minimum Variance (GMV).

Implements:
  1. ERC (Equal Risk Contribution)
     - Multiplicative update (Maillard / Spinu style)
     - SLSQP on variance of risk contributions (long-only, optional floors)
  2. Inverse-vol risk parity (diagonal ERC approximation)
  3. Global Minimum Variance
     - Analytical unconstrained (Σ^{-1} 1)
     - Long-only SLSQP
     - Long-only with per-name caps (uniform cap, default 5%)

Outputs:
  erc_gmv_strategies.csv       — weights + risk contributions by strategy
  erc_gmv_summary.csv          — portfolio vol / return / RC dispersion

Usage:
  python portfolio_optimization.py
  python portfolio_optimization.py --window 126 --name-cap 0.05
  python portfolio_optimization.py --universe growth_ai
  python maintain_analytics.py optimize
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cli_common import (
    add_index_args, add_ticker_args, add_sector_arg, add_save_arg,
    add_window_arg, resolve_tickers_from_args, resolve_index_names_from_args,
    build_parser,
)
from index_registry import parse_indexes, tickers_for_index, available_indexes, index_help_text

from scipy.optimize import minimize  # noqa: F401  (canonical availability flag in analytics_common)
from analytics_common import HAS_SCIPY  # canonical scipy-availability flag

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
HOLDINGS = DATA_DIR / "portfolio_holdings.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
CALENDAR = DATA_DIR / "rebalance_calendar.csv"
HMM = DATA_DIR / "hmm_regime_states.csv"
OUT_W = DATA_DIR / "erc_gmv_strategies.csv"
OUT_S = DATA_DIR / "erc_gmv_summary.csv"


def load_returns(tickers: list[str], window: int = 126) -> pd.DataFrame:
    prices = pd.read_parquet(PRICES)
    prices["date"] = pd.to_datetime(prices["date"])
    wide = (
        prices[prices["ticker"].isin(tickers)]
        .pivot_table(index="date", columns="ticker", values="close")
        .sort_index()
        .ffill()
    )
    rets = np.log(wide / wide.shift(1)).dropna(how="all").iloc[-window:]
    rets = rets.dropna(axis=1, thresh=max(40, window // 3))
    return rets


def stats(w: np.ndarray, cov: np.ndarray, mu: np.ndarray | None = None) -> dict:
    w = np.asarray(w, float)
    w = w / w.sum()
    sig2 = float(w @ cov @ w)
    sig = float(np.sqrt(max(sig2, 0.0)))
    mrc = cov @ w
    rc = w * mrc
    out = {
        "vol": sig,
        "rc": rc,
        "mrc": mrc,
        "rc_pct": rc / sig2 if sig2 > 1e-18 else rc,
        "rc_dispersion": float(np.std(rc / sig2)) if sig2 > 1e-18 else float("nan"),
    }
    if mu is not None:
        out["ret"] = float(w @ mu)
    return out


# ---------------------------------------------------------------------------
# ERC
# ---------------------------------------------------------------------------

def erc_multiplicative(cov: np.ndarray, max_iter: int = 5000, tol: float = 1e-14) -> tuple[np.ndarray, int]:
    n = cov.shape[0]
    w = np.ones(n) / n
    for i in range(max_iter):
        sig2 = float(w @ cov @ w)
        rc = w * (cov @ w)
        adj = (sig2 / n) / np.maximum(rc, 1e-18)
        w_new = w * adj
        w_new = w_new / w_new.sum()
        if float(np.max(np.abs(w_new - w))) < tol:
            return w_new, i + 1
        w = w_new
    return w, max_iter


def erc_slsqp(cov: np.ndarray, w_floor: float = 0.01) -> tuple[np.ndarray, bool]:
    """
    Long-only ERC: minimize variance of risk contributions with weight floor
    so the solution stays interior (all names participate unless infeasible).
    """
    n = cov.shape[0]
    if not HAS_SCIPY:
        w, _ = erc_multiplicative(cov)
        return w, False

    def obj(w):
        w = np.asarray(w, float)
        sig2 = float(w @ cov @ w)
        if sig2 <= 1e-18:
            return 1e6
        rc = w * (cov @ w)
        target = sig2 / n
        return float(np.sum((rc - target) ** 2))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(w_floor, 1.0)] * n
    # if n * floor > 1, relax floor
    if n * w_floor > 1.0:
        bounds = [(0.0, 1.0)] * n
        w_floor = 0.0
    w0 = np.ones(n) / n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 2000, "ftol": 1e-16, "disp": False})
    w = np.maximum(res.x, 0.0)
    if w.sum() <= 0:
        w = np.ones(n) / n
    else:
        w = w / w.sum()
    return w, bool(res.success)


def inv_vol_weights(cov: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    w = 1.0 / vols
    return w / w.sum()


# ---------------------------------------------------------------------------
# GMV
# ---------------------------------------------------------------------------

def gmv_unconstrained(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    inv = np.linalg.pinv(cov)
    w = inv @ np.ones(n)
    return w / w.sum()


def gmv_long_only(cov: np.ndarray) -> tuple[np.ndarray, bool]:
    n = cov.shape[0]
    if HAS_SCIPY:
        def obj(w):
            return float(w @ cov @ w)
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        res = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 2000, "ftol": 1e-16})
        w = np.maximum(res.x, 0.0)
        return w / w.sum(), bool(res.success)
    # projected gradient fallback
    w = np.ones(n) / n
    for _ in range(800):
        w = w - 0.05 * (2 * cov @ w)
        w = np.maximum(w, 0.0)
        s = w.sum()
        w = (np.ones(n) / n) if s <= 0 else (w / s)
    return w, True


def gmv_long_capped(cov: np.ndarray, tickers: list[str], caps: dict[str, float]) -> tuple[np.ndarray, bool]:
    n = cov.shape[0]
    cap = np.array([caps.get(t, 1.0) for t in tickers], float)
    if HAS_SCIPY and cap.sum() >= 1.0 - 1e-9:
        def obj(w):
            return float(w @ cov @ w)
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, float(cap[i])) for i in range(n)]
        w0 = np.minimum(np.ones(n) / n, cap)
        w0 = w0 / w0.sum()
        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 2000, "ftol": 1e-16})
        w = np.maximum(res.x, 0.0)
        return w / w.sum(), bool(res.success)
    w, ok = gmv_long_only(cov)
    for _ in range(80):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap[over]).sum())
        w[over] = cap[over]
        free = ~over & (w < cap - 1e-12)
        if free.any() and w[free].sum() > 0:
            w[free] += excess * (w[free] / w[free].sum())
        w = np.maximum(w, 0.0)
        w = w / w.sum()
    return w, ok


def vol_target_renorm(cov: np.ndarray, tickers: list[str], target: float = 0.25,
                      name_cap: float = 0.05) -> np.ndarray:
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    raw = []
    for i, t in enumerate(tickers):
        cap = name_cap
        raw.append(float(np.clip(target / vols[i], 0.0, cap)))
    w = np.array(raw)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Universe + driver
# ---------------------------------------------------------------------------

def resolve_universe(name: str) -> list[str]:
    try:
        names = parse_indexes(name)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    seen, out = set(), []
    for n in names:
        for tk in tickers_for_index(n):
            if tk not in seen:
                seen.add(tk)
                out.append(tk)
    if not out:
        raise SystemExit(f"No tickers for universe={name!r}. Available: {available_indexes()}")
    return out



def current_weights(tickers: list[str]) -> np.ndarray:
    if not HOLDINGS.exists():
        return np.ones(len(tickers)) / len(tickers)
    h = pd.read_parquet(HOLDINGS)
    cw = h.set_index("ticker")["weight"].astype(float)
    if cw.sum() > 2:
        cw = cw / 100.0
    w = np.array([float(cw.get(t, 0.0)) for t in tickers])
    if w.sum() <= 0:
        return np.ones(len(tickers)) / len(tickers)
    return w / w.sum()


def turnover_band() -> float:
    """Return the latest turnover_band from rebalance_calendar.csv, or 1.0 if unavailable."""
    if not CALENDAR.exists():
        return 1.0
    cal = pd.read_csv(CALENDAR)
    if cal.empty:
        return 1.0
    date_col = "rebalance_date" if "rebalance_date" in cal.columns else ("date" if "date" in cal.columns else None)
    if date_col is None or "turnover_band" not in cal.columns:
        return 1.0
    cal[date_col] = pd.to_datetime(cal[date_col]).dt.date
    today = date.today()
    row = cal[cal[date_col] <= today].tail(1)
    if row.empty:
        return 1.0
    return float(row.iloc[-1]["turnover_band"])


def apply_turnover_cap(w: np.ndarray, w_cur: np.ndarray, band: float) -> np.ndarray:
    """Cap weight drift to +/- band * w_cur per ticker, then renormalize."""
    if band >= 1.0:
        return w
    lower = np.maximum(w_cur - band * w_cur, 0.0)
    upper = w_cur + band * w_cur
    w_capped = np.clip(w, lower, upper)
    s = w_capped.sum()
    return w_capped / s if s > 0 else w


def regime_scaled_cov(cov: np.ndarray, tickers: list[str], window: int = 126) -> np.ndarray:
    """Blend sample covariance toward a crisis-elevated correlation matrix.

    Measured regime shift: avg pairwise corr ~0.19 calm vs ~0.40 crisis
    (crisis_correlation.py). When HMM says high_vol_stress, correlations are
    scaled up toward the crisis level so allocation doesn't assume the calm
    regime will persist. Blend weight = 0.5 in stress, 0.1 otherwise.
    """
    import numpy as np

    reg = ""
    if HMM.exists():
        try:
            h = pd.read_csv(HMM)
            h["date"] = pd.to_datetime(h.get("date"), errors="coerce")
            h = h.dropna(subset=["date"]).sort_values("date")
            if len(h):
                for c in ("regime", "state", "label"):
                    if c in h.columns:
                        reg = str(h.iloc[-1][c])
                        break
        except Exception:  # noqa: BLE001
            reg = ""
    stress = "stress" in reg.lower()
    # crisis correlation matrix from 63d sample (the shift is in corr, not vol)
    rets = load_returns(tickers, window=window)
    corr63 = np.array(rets.corr().values, copy=True)
    np.fill_diagonal(corr63, 1.0)
    # shift the average off-diagonal correlation toward the measured crisis
    # level (0.40) rather than multiply — multiplicative inflation blows up
    # already-high pairs past 1.0 and distorts the cross-section.
    calm_level, crisis_level = 0.19, 0.40
    n = len(tickers)
    off_mask = ~np.eye(n, dtype=bool)
    cur_mean = float(corr63[off_mask].mean()) if n > 1 else calm_level
    delta = crisis_level - cur_mean
    shifted = np.where(off_mask, np.clip(corr63 + delta, -1, 1), 1.0).reshape(corr63.shape)
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    cov_crisis = (shifted * vols) * vols[:, None]
    blend = 0.5 if stress else 0.1
    return blend * cov_crisis + (1 - blend) * cov


def run(universe: str = "portfolio", window: int = 126, name_cap: float = 0.05,
        w_floor: float = 0.02) -> None:
    tickers = resolve_universe(universe)
    rets = load_returns(tickers, window=window)
    tickers = list(rets.columns)
    cov = rets.cov().values * 252.0
    mu = rets.mean().values * 252.0
    cov = regime_scaled_cov(cov, tickers, window=window)

    w_cur = current_weights(tickers)
    band = turnover_band()
    w_erc_m, iters = erc_multiplicative(cov)
    w_erc_m = apply_turnover_cap(w_erc_m, w_cur, band)
    w_erc_s, ok_erc = erc_slsqp(cov, w_floor=w_floor)
    w_erc_s = apply_turnover_cap(w_erc_s, w_cur, band)
    w_iv = inv_vol_weights(cov)
    w_iv = apply_turnover_cap(w_iv, w_cur, band)
    w_gmv_u = gmv_unconstrained(cov)
    w_gmv_u = apply_turnover_cap(w_gmv_u, w_cur, band)
    w_gmv_l, ok_g = gmv_long_only(cov)
    w_gmv_l = apply_turnover_cap(w_gmv_l, w_cur, band)
    caps = {t: name_cap for t in tickers}
    w_gmv_c, ok_c = gmv_long_capped(cov, tickers, caps)
    w_gmv_c = apply_turnover_cap(w_gmv_c, w_cur, band)
    w_vt = vol_target_renorm(cov, tickers, name_cap=name_cap)
    w_vt = apply_turnover_cap(w_vt, w_cur, band)

    strategies = {
        "Current": w_cur,
        "ERC_multiplicative": w_erc_m,
        "ERC_SLSQP_floor": w_erc_s,
        "InvVol_RP": w_iv,
        "GMV_unconstrained": w_gmv_u,
        "GMV_long_only": w_gmv_l,
        "GMV_long_capped": w_gmv_c,
        "VolTarget_renorm": w_vt,
    }

    # weights table (wide)
    print(f"Universe={universe}  n={len(tickers)}  window={window}d  scipy={HAS_SCIPY}")
    print(f"ERC multiplicative iters={iters}  ERC SLSQP ok={ok_erc}  GMV ok={ok_g}/{ok_c}")
    print("\n=== Weights % ===")
    rows_w = []
    rows_long = []
    rows_sum = []
    for name, w in strategies.items():
        st = stats(w, cov, mu)
        row = {"strategy": name, "universe": universe}
        for i, t in enumerate(tickers):
            row[t] = round(100.0 * float(w[i]), 2)
            rows_long.append({
                "universe": universe,
                "strategy": name,
                "ticker": t,
                "weight": float(w[i]),
                "rc_pct_var": float(st["rc_pct"][i]),
                "sigma_i": float(np.sqrt(cov[i, i])),
            })
        row["port_vol"] = round(100.0 * st["vol"], 2)
        row["port_ret"] = round(100.0 * st["ret"], 2)
        row["rc_dispersion"] = round(100.0 * st["rc_dispersion"], 3)
        rows_w.append(row)
        rows_sum.append({
            "universe": universe,
            "strategy": name,
            "port_vol": st["vol"],
            "port_ret": st["ret"],
            "rc_dispersion": st["rc_dispersion"],
            "n_names": int((w > 1e-4).sum()),
            "max_weight": float(w.max()),
        })

    dfw = pd.DataFrame(rows_w)
    cols = ["strategy"] + tickers + ["port_vol", "port_ret", "rc_dispersion"]
    print(dfw[cols].to_string(index=False))

    print("\n=== Risk contribution % of portfolio variance ===")
    for name in ["Current", "ERC_SLSQP_floor", "InvVol_RP", "GMV_long_only", "GMV_long_capped"]:
        st = stats(strategies[name], cov)
        rc = {t: round(100.0 * float(st["rc_pct"][i]), 1) for i, t in enumerate(tickers)}
        print(f"  {name:22s} σ={st['vol']*100:5.2f}%  RC={rc}")

    pd.DataFrame(rows_long).to_csv(OUT_W, index=False)
    pd.DataFrame(rows_sum).to_csv(OUT_S, index=False)
    print(f"\nWrote {OUT_W}")
    print(f"Wrote {OUT_S}")


def main():
    ap = argparse.ArgumentParser(description="ERC risk parity & minimum variance")
    add_index_args(ap, default="portfolio")
    ap.add_argument("--window", type=int, default=126)
    ap.add_argument("--name-cap", type=float, default=0.05)
    ap.add_argument("--w-floor", type=float, default=0.02, help="ERC SLSQP minimum weight")
    args = ap.parse_args()
    idxs = resolve_index_names_from_args(args, default_index='portfolio')
    uni = ','.join(idxs) if idxs else 'portfolio'
    run(universe=uni, window=args.window, name_cap=args.name_cap, w_floor=args.w_floor)


if __name__ == "__main__":
    main()
