#!/usr/bin/env python3
"""
monte_carlo.py — Regime-switching Monte Carlo with variance-reduction options.

Uses HMM transition matrix + regime-conditional return moments estimated from
daily_prices (and optional portfolio / index membership via index_registry).

Variance reduction:
  - antithetic   : pair Z and -Z (same regime path or paired shocks)
  - control      : control variate = equal-weight market path with known analytic mean
  - stratified   : stratify initial regime by stationary distribution
  - quasi        : Sobol quasi-Monte Carlo for Gaussian shocks (optional scipy/scipy.stats
                   or numpy fallback)

Usage:
  python monte_carlo.py --index portfolio --n-paths 5000 --horizon 63 --save
  python monte_carlo.py --ticker MOS,PFE,T --n-paths 2000 --vr antithetic,control --save
  python monte_carlo.py --index dual --n-paths 3000 --vr all --horizon 126 --save
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
HMM_STATES = DATA_DIR / "hmm_regime_states.csv"
HMM_TRANS = DATA_DIR / "hmm_transition_matrix.csv"
OUT_SUMMARY = DATA_DIR / "monte_carlo_summary.csv"
OUT_PATHS = DATA_DIR / "monte_carlo_path_stats.csv"
OUT_WEALTH = DATA_DIR / "monte_carlo_terminal_wealth.csv"

REGIME_ORDER_DEFAULT = ["low_vol", "normal", "high_vol_stress"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_returns(tickers: list[str] | None = None) -> pd.DataFrame:
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    if tickers:
        prices = prices[prices["ticker"].isin(tickers)]
    wide = (
        prices.pivot_table(index="date", columns="ticker", values="close")
        .sort_index()
        .ffill()
    )
    rets = np.log(wide / wide.shift(1)).dropna(how="all")
    return rets


def load_hmm_alignment(rets_index: pd.DatetimeIndex) -> pd.Series:
    if not HMM_STATES.exists():
        raise SystemExit("Missing hmm_regime_states.csv — run hmm_regime_detection.py --save")
    h = pd.read_csv(HMM_STATES)
    h["date"] = pd.to_datetime(h["date"])
    s = h.set_index("date")["regime"].reindex(rets_index).ffill().bfill()
    return s


def load_transition(regimes: list[str]) -> np.ndarray:
    """Return P[i,j] = P(to=regimes[j] | from=regimes[i])."""
    if not HMM_TRANS.exists():
        # sticky default
        n = len(regimes)
        P = np.full((n, n), 0.05 / max(n - 1, 1))
        np.fill_diagonal(P, 0.95)
        return P
    tm = pd.read_csv(HMM_TRANS, index_col=0)
    # align to regimes order
    P = np.zeros((len(regimes), len(regimes)))
    for i, a in enumerate(regimes):
        for j, b in enumerate(regimes):
            if a in tm.index and b in tm.columns:
                P[i, j] = float(tm.loc[a, b])
            elif a in tm.columns and b in tm.index:
                P[i, j] = float(tm.loc[b, a])  # unlikely
    # row-normalize
    row_sum = P.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    P = P / row_sum
    return P


def stationary_dist(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P for eigenvalue 1."""
    w, v = np.linalg.eig(P.T)
    i = np.argmin(np.abs(w - 1.0))
    vec = np.real(v[:, i])
    vec = np.maximum(vec, 0)
    s = vec.sum()
    return vec / s if s > 0 else np.ones(len(vec)) / len(vec)


def regime_moments(
    rets: pd.DataFrame, regimes: pd.Series, regime_names: list[str]
) -> dict[str, dict]:
    """Per-regime mean vector and covariance (daily log returns)."""
    out = {}
    for name in regime_names:
        mask = regimes == name
        block = rets.loc[mask].dropna(how="all")
        if len(block) < 5:
            # fallback to global
            block = rets.dropna(how="all")
        mu = block.mean().values.astype(float)
        cov = block.cov().values.astype(float)
        # ridge for PSD
        cov = 0.5 * (cov + cov.T)
        eig = np.linalg.eigvalsh(cov)
        if eig.min() < 1e-10:
            cov = cov + np.eye(cov.shape[0]) * (1e-8 - min(eig.min(), 0))
        out[name] = {
            "mu": mu,
            "cov": cov,
            "n": int(mask.sum()),
            "columns": list(rets.columns),
        }
    return out


def chol_or_diag(cov: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # eigenvalue clip
        w, v = np.linalg.eigh(cov)
        w = np.maximum(w, 1e-10)
        return v @ np.diag(np.sqrt(w))


# ---------------------------------------------------------------------------
# Shock generation + variance reduction
# ---------------------------------------------------------------------------

def draw_gaussian(n_paths: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n_paths, dim))


def antithetic_shocks(Z: np.ndarray) -> np.ndarray:
    """Stack Z and -Z → 2n paths."""
    return np.vstack([Z, -Z])


def sobol_gaussian(n_paths: int, dim: int, seed: int = 0) -> np.ndarray:
    """Owen-scrambled Sobol → Gaussian (see sobol_qmc.py)."""
    try:
        from sobol_qmc import sobol_normal
        return sobol_normal(n_paths, dim, seed=seed)
    except Exception:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_paths, dim))


def control_variate_adjust(
    samples: np.ndarray,
    control: np.ndarray,
    control_expect: float,
) -> np.ndarray:
    """θ_hat = X - β(C - E[C]), β = Cov(X,C)/Var(C)."""
    x = samples.astype(float)
    c = control.astype(float)
    vc = np.var(c)
    if vc < 1e-16:
        return x
    beta = np.cov(x, c)[0, 1] / vc
    return x - beta * (c - control_expect)


# ---------------------------------------------------------------------------
# Regime path simulation
# ---------------------------------------------------------------------------

def simulate_regimes(
    P: np.ndarray,
    n_paths: int,
    horizon: int,
    start_probs: np.ndarray,
    rng: np.random.Generator,
    stratified: bool = False,
) -> np.ndarray:
    """
    Returns regime index array shape (n_paths, horizon).
    If stratified, allocate initial regimes proportional to start_probs.
    """
    n_states = P.shape[0]
    paths = np.zeros((n_paths, horizon), dtype=int)

    if stratified:
        counts = np.floor(start_probs * n_paths).astype(int)
        while counts.sum() < n_paths:
            counts[np.argmax(start_probs)] += 1
        while counts.sum() > n_paths:
            counts[np.argmax(counts)] -= 1
        init = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
        rng.shuffle(init)
        paths[:, 0] = init[:n_paths]
    else:
        paths[:, 0] = rng.choice(n_states, size=n_paths, p=start_probs)

    # CDF of each row for inverse-transform
    cdf = np.cumsum(P, axis=1)
    for t in range(1, horizon):
        u = rng.random(n_paths)
        prev = paths[:, t - 1]
        for i in range(n_paths):
            paths[i, t] = int(np.searchsorted(cdf[prev[i]], u[i], side="right"))
    return paths


def simulate_returns(
    regime_paths: np.ndarray,
    moments: dict[str, dict],
    regime_names: list[str],
    Z: np.ndarray,
) -> np.ndarray:
    """
    regime_paths: (n_paths, horizon)
    Z: (n_paths, horizon, n_assets) standard normal
    returns: (n_paths, horizon, n_assets) log returns
    """
    n_paths, horizon = regime_paths.shape
    n_assets = Z.shape[2]
    out = np.zeros((n_paths, horizon, n_assets))
    L_cache = {}
    mu_cache = {}
    for name in regime_names:
        mu_cache[name] = moments[name]["mu"]
        L_cache[name] = chol_or_diag(moments[name]["cov"])

    for t in range(horizon):
        for i in range(n_paths):
            name = regime_names[regime_paths[i, t]]
            out[i, t] = mu_cache[name] + L_cache[name] @ Z[i, t]
    return out


def wealth_paths(
    log_rets: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    log_rets: (n_paths, horizon, n_assets)
    weights: (n_assets,) or None → equal weight
    Returns wealth (n_paths, horizon+1) starting at 1.0
    """
    n_paths, horizon, n_assets = log_rets.shape
    if weights is None:
        weights = np.ones(n_assets) / n_assets
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    # portfolio simple return ≈ sum w * (exp(r)-1); use log-sum for stability via weighted exp
    port = np.tensordot(np.exp(log_rets) - 1.0, weights, axes=([2], [0]))  # (n_paths, horizon)
    wealth = np.ones((n_paths, horizon + 1))
    for t in range(horizon):
        wealth[:, t + 1] = wealth[:, t] * (1.0 + port[:, t])
    return wealth


def path_stats(wealth: np.ndarray) -> dict[str, float]:
    terminal = wealth[:, -1]
    # max drawdown per path
    peak = np.maximum.accumulate(wealth, axis=1)
    dd = wealth / peak - 1.0
    max_dd = dd.min(axis=1)
    return {
        "mean_terminal": float(terminal.mean()),
        "median_terminal": float(np.median(terminal)),
        "p05_terminal": float(np.quantile(terminal, 0.05)),
        "p25_terminal": float(np.quantile(terminal, 0.25)),
        "p75_terminal": float(np.quantile(terminal, 0.75)),
        "p95_terminal": float(np.quantile(terminal, 0.95)),
        "std_terminal": float(terminal.std()),
        "mean_max_dd": float(max_dd.mean()),
        "p05_max_dd": float(np.quantile(max_dd, 0.05)),
        "prob_loss": float((terminal < 1.0).mean()),
        "prob_gain_10pct": float((terminal > 1.10).mean()),
        "n_paths": int(len(terminal)),
    }


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def run_regime_mc(
    tickers: list[str],
    n_paths: int = 4000,
    horizon: int = 63,
    seed: int = 42,
    vr: Iterable[str] | None = None,
    weights: np.ndarray | None = None,
) -> dict:
    """
    vr options: antithetic, control, stratified, quasi, all
    """
    vr_set = set()
    if vr:
        # Accept str "a,b" or list/tuple of tokens
        if isinstance(vr, str):
            tokens = vr.split(",")
        else:
            tokens = []
            for v in vr:
                tokens.extend(str(v).split(","))
        for part in tokens:
            part = part.strip().lower()
            if part == "all":
                vr_set.update(["antithetic", "control", "stratified", "quasi"])
            elif part and part != "none":
                vr_set.add(part)

    rets = load_returns(tickers)
    # drop columns all-nan
    rets = rets.dropna(axis=1, how="all").dropna(how="all")
    cols = list(rets.columns)
    if not cols:
        raise SystemExit("No return data for requested tickers")

    regimes_s = load_hmm_alignment(rets.index)
    # regime names present in data, prefer default order
    present = [r for r in REGIME_ORDER_DEFAULT if (regimes_s == r).any()]
    for r in regimes_s.dropna().unique():
        if r not in present:
            present.append(r)
    regime_names = present
    P = load_transition(regime_names)
    pi0 = stationary_dist(P)
    moments = regime_moments(rets, regimes_s, regime_names)
    n_assets = len(cols)

    rng = np.random.default_rng(seed)
    stratified = "stratified" in vr_set
    use_quasi = "quasi" in vr_set
    use_anti = "antithetic" in vr_set
    use_control = "control" in vr_set

    # Base path count before antithetic doubling
    base_n = n_paths // 2 if use_anti else n_paths
    base_n = max(base_n, 1)

    regime_paths = simulate_regimes(
        P, base_n, horizon, pi0, rng, stratified=stratified
    )
    if use_anti:
        # same regimes for antithetic pair (common random regimes, antithetic shocks)
        regime_paths = np.vstack([regime_paths, regime_paths])

    # Shocks
    if use_quasi:
        Z_flat = sobol_gaussian(regime_paths.shape[0] * horizon, n_assets, seed=seed)
        Z = Z_flat.reshape(regime_paths.shape[0], horizon, n_assets)
        if use_anti:
            half = regime_paths.shape[0] // 2
            Z[half:] = -Z[:half]
    else:
        Z = rng.standard_normal((regime_paths.shape[0], horizon, n_assets))
        if use_anti:
            half = regime_paths.shape[0] // 2
            Z[half:] = -Z[:half]

    log_rets = simulate_returns(regime_paths, moments, regime_names, Z)
    wealth = wealth_paths(log_rets, weights=weights)

    terminal = wealth[:, -1]
    control_stats = {}
    if use_control:
        # Control variate = path-specific *drift-only* wealth under the same regimes.
        # C_k = Π_t (1 + w' (exp(μ_{s_{k,t}}) - 1));  E[C] ≈ mean_k C_k under π₀,P
        # (using sample mean of C is slightly biased for β but standard in practice;
        #  we use a mixture-drift closed form for E[C] instead).
        if weights is None:
            w = np.ones(n_assets) / n_assets
        else:
            w = np.asarray(weights, dtype=float)
            w = w / w.sum()
        drift_log = np.zeros((regime_paths.shape[0], horizon))
        for i, name in enumerate(regime_names):
            mu = moments[name]["mu"]
            drift_log[:, :] = np.where(
                regime_paths == i,
                float(w @ mu),
                drift_log,
            )
        # vectorized: for each t, map regime to drift
        drift_log = np.zeros((regime_paths.shape[0], horizon))
        mu_w = {name: float(w @ moments[name]["mu"]) for name in regime_names}
        for i, name in enumerate(regime_names):
            drift_log = np.where(regime_paths == i, mu_w[name], drift_log)
        control_path = np.cumprod(np.exp(drift_log), axis=1)  # start implicit 1 at t0
        C = control_path[:, -1]
        # E[C] under stationary regime mixture (constant drift approx)
        mix_mu = sum(float(pi0[i]) * mu_w[name] for i, name in enumerate(regime_names))
        control_expect = float(np.exp(mix_mu * horizon))
        raw_std = float(terminal.std())
        terminal_adj = control_variate_adjust(terminal, C, control_expect)
        control_stats = {
            "control_expect": control_expect,
            "raw_std": raw_std,
            "adj_std": float(terminal_adj.std()),
            "vr_ratio": float(raw_std / max(float(terminal_adj.std()), 1e-12)),
        }
        terminal = terminal_adj
        wealth = wealth.copy()
        wealth[:, -1] = terminal

    stats = path_stats(wealth)
    stats.update({
        "horizon": horizon,
        "n_assets": n_assets,
        "tickers": ",".join(cols),
        "vr": ",".join(sorted(vr_set)) if vr_set else "none",
        "seed": seed,
        "regimes": ",".join(regime_names),
    })
    stats.update({f"control_{k}": v for k, v in control_stats.items()})

    # regime occupancy
    for i, name in enumerate(regime_names):
        stats[f"pct_time_{name}"] = float((regime_paths == i).mean())

    return {
        "stats": stats,
        "terminal": terminal,
        "wealth": wealth,
        "regime_paths": regime_paths,
        "regime_names": regime_names,
        "columns": cols,
        "P": P,
        "pi0": pi0,
    }


def resolve_tickers(args) -> list[str]:
    try:
        from cli_common import resolve_tickers_from_args
        return resolve_tickers_from_args(args, default_index="portfolio")
    except Exception:
        from index_registry import tickers_for_index, parse_indexes
        if getattr(args, "ticker", None):
            return [x.strip().upper() for x in args.ticker.split(",") if x.strip()]
        names = parse_indexes(getattr(args, "index", None) or "portfolio")
        out = []
        for n in names:
            out.extend(tickers_for_index(n))
        return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser(description="Regime-switching Monte Carlo")
    try:
        from cli_common import add_index_args, add_ticker_args, add_save_arg
        add_index_args(ap, default="portfolio")
        add_ticker_args(ap)
        add_save_arg(ap)
    except Exception:
        ap.add_argument("--index", action="append", default=None)
        ap.add_argument("--ticker", default=None)
        ap.add_argument("--save", action="store_true")
    ap.add_argument("--n-paths", type=int, default=4000)
    ap.add_argument("--horizon", type=int, default=63, help="Trading days")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--vr",
        default="antithetic,control",
        help="Variance reduction: antithetic,control,stratified,quasi,all,none",
    )
    args = ap.parse_args()

    tickers = resolve_tickers(args)
    if not tickers:
        raise SystemExit("No tickers resolved")
    print(f"Regime MC  tickers={len(tickers)}  n_paths={args.n_paths}  horizon={args.horizon}  vr={args.vr}")

    vr = None if args.vr.strip().lower() == "none" else args.vr
    result = run_regime_mc(
        tickers,
        n_paths=args.n_paths,
        horizon=args.horizon,
        seed=args.seed,
        vr=vr,
    )
    stats = result["stats"]
    print("\n=== Terminal wealth (start=1.0) ===")
    for k in [
        "mean_terminal", "median_terminal", "p05_terminal", "p95_terminal",
        "std_terminal", "mean_max_dd", "prob_loss", "prob_gain_10pct",
        "vr", "control_vr_ratio",
    ]:
        if k in stats:
            print(f"  {k:20s}  {stats[k]}")
    print("\n=== Regime occupancy (simulated) ===")
    for k, v in stats.items():
        if k.startswith("pct_time_"):
            print(f"  {k:20s}  {v:.3f}")

    print("\n=== Transition matrix ===")
    print(pd.DataFrame(result["P"], index=result["regime_names"], columns=result["regime_names"]).round(3))

    if args.save:
        pd.DataFrame([stats]).to_csv(OUT_SUMMARY, index=False)
        # path quantiles over time
        w = result["wealth"]
        q = np.quantile(w, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        path_df = pd.DataFrame({
            "t": np.arange(w.shape[1]),
            "p05": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4],
            "mean": w.mean(axis=0),
        })
        path_df.to_csv(OUT_PATHS, index=False)
        pd.DataFrame({"terminal_wealth": result["terminal"]}).to_csv(OUT_WEALTH, index=False)
        print(f"\nWrote {OUT_SUMMARY}\nWrote {OUT_PATHS}\nWrote {OUT_WEALTH}")


if __name__ == "__main__":
    main()
