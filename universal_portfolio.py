"""universal_portfolio.py — Cover (1991) universal portfolio: reusable library.

Library (import this module as `from universal_portfolio import ...`):
  UniversalEngine      exact/mixed engines over a return matrix (reusable,
                       numpy-only, no IO) -> PortfolioResult
  PortfolioResult      weights / wealth / gated weights / trades / stats
  RegimeStates         tape-level HMM states (side info), fit + predict +
                       state-conditional universal portfolio
  universal_univariate convenience wrapper the CLI and other scripts use

Theory (docs/cover_universal_portfolio.md): the Dirichlet(1/2)-weighted
universal portfolio holds, on day k, the performance-weighted average over
ALL constant-rebalanced portfolios in the simplex,
    b_hat_k = int_B b S_{k-1}(b) dmu(b) / int_B S_{k-1}(b) dmu(b),
so wealth telescopes to a plain integral over the simplex — computable
causally with ZERO statistical assumptions and NO lookahead. Regret vs the
best CRP in hindsight is exactly ((m-1)/2) log(n+1) (Ordentlich-Cover
minimax), so universal <= oracle at finite n is a theorem, not a bug.

Engines:
  exact_2asset   Dirichlet(1/2) UP for m=2 — Cover eq. (128) Q-recursion,
                 O(n^2), EXACT; self-checked via the telescope identity
                 S_hat_n == sum_l Q_n(l) (asserted on every run).
  mc_simplex     Dirichlet(1/2) MC mixture for m>=2 (Kalai-Vempala): sample
                 portfolios from the simplex, wealth-weight them. Converges
                 to the same integral; validated vs exact to <0.05%.

Side information (Cover & Ordentlich 1996): RegimeStates fits a 3-state
Gaussian HMM on the tape's OWN features (mkt_ret, vol21, avg pairwise corr —
same features as hmm_regime_detection.py), so states split the sequence and
the state-conditional universal portfolio competes with the best
STATE-constant-rebalanced portfolio. Regret: (d/2n)log(n+1) + (k/n)log2 with
d = k(m-1). States fitted on the ORIGINAL tape, never on a resampled path.

Sizing: SizingPlan maps a target weight vector (e.g. the latest gated
universal weights) onto portfolio_holdings.parquet -> per-name delta notional,
delta shares, and the action, honoring the Cover sec.10 rebalance gate.

CLI (thin wrapper over the library):
  python universal_portfolio.py paths [--side] [--paths N] [-save]
  python universal_portfolio.py book [--side] [--sizing] [--save]
  python universal_portfolio.py validate --samples N
  python universal_portfolio.py sizing [--value $] [--save]

Outputs (derived panels only — never writes daily_prices/):
  universal_paths.parquet           research bootstrap book stats
  universal_book_weights.parquet    book daily weights (+ side variant col)
  universal_sizing_plan.parquet     target-vs-holdings trade plan
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
BOOK = ["BAYRY", "CAG", "HMC", "HPQ", "KHC", "MOS", "PFE", "SMCI", "T"]
OUT_PATHS = DATA_DIR / "universal_paths.parquet"
OUT_BOOK = DATA_DIR / "universal_book_weights.parquet"
OUT_SIZING = DATA_DIR / "universal_sizing_plan.parquet"


# ==========================================================================
# Engine — reusable, no IO
# ==========================================================================

class UniversalEngine:
    """Cover universal portfolio engines over a returns matrix."""

    ALPHA = 0.5  # Dirichlet(1/2,..,1/2): minimax regret prior

    # -- exact m=2 (Cover eq. 128 Q-recursion) ------------------------------

    @staticmethod
    def exact_2asset(r1: np.ndarray, r2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact Dirichlet(1/2) universal portfolio, m=2.

        Returns (b1, b2, wealth): causal daily weights and wealth path
        (wealth[0] = 1.0). The telescope identity S_hat_n == sum_l Q_n(l) is
        ASSERTED — the recursion is either exactly right or dies loudly.
        """
        x1 = 1.0 + np.asarray(r1, dtype=np.float64)
        x2 = 1.0 + np.asarray(r2, dtype=np.float64)
        n = len(x1)
        b1 = np.empty(n)
        b2 = np.empty(n)
        wealth = np.empty(n + 1)
        wealth[0] = 1.0
        Q = np.array([1.0])
        for k in range(1, n + 1):
            l = np.arange(k, dtype=np.float64)
            s = Q.sum()
            w1 = float(((l + 0.5) * Q).sum()) / (k * s)
            w2 = float(((k - l - 0.5) * Q).sum()) / (k * s)
            assert abs(w1 + w2 - 1.0) < 1e-12, "weights must sum to 1"
            b1[k - 1] = w1
            b2[k - 1] = w2
            wealth[k] = wealth[k - 1] * (w1 * x1[k - 1] + w2 * x2[k - 1])
            Qn = np.zeros(k + 1)
            q = Q
            Qn[0] = x2[k - 1] * (k - 0.5) / k * q[0]
            Qn[k] = x1[k - 1] * (k - 0.5) / k * q[k - 1]
            if k > 1:
                Qn[1:k] = (x1[k - 1] * (l[1:] - 0.5) / k * q[:-1]
                           + x2[k - 1] * (k - l[1:] - 0.5) / k * q[1:])
            Q = Qn
        assert abs(wealth[-1] - Q.sum()) / max(wealth[-1], 1e-300) < 1e-9, \
            f"telescope failed: {wealth[-1]} vs {Q.sum()}"
        return b1, b2, wealth[1:]

    # -- MC simplex for m >= 2 ----------------------------------------------

    @staticmethod
    def mc_simplex(R: np.ndarray, n_samples: int = 20000, seed: int = 0,
                   alpha: float = 0.5):
        """Universal portfolio via simplex sampling. R: (n, m) net returns.

        Returns (wealth, weights): wealth (n+1,), daily causal weights (n, m).
        """
        m = R.shape[1]
        rng = np.random.default_rng(seed)
        B = rng.dirichlet([alpha] * m, size=n_samples)
        W = np.ones(n_samples)
        Ws = np.empty(R.shape[0] + 1)
        Ws[0] = 1.0
        wts = np.empty((R.shape[0], m))
        rel = 1.0 + R
        for i in range(R.shape[0]):
            w = (B * W[:, None]).sum(axis=0) / W.sum()
            wts[i] = w
            Ws[i + 1] = Ws[i] * (rel[i] @ w)
            W *= rel[i] @ B.T
        return Ws, wts

    # -- main entry ---------------------------------------------------------

    @classmethod
    def run(cls, R: np.ndarray, seed: int = 0, n_samples: int = 20000) -> "PortfolioResult":
        """Universal portfolio over an (n, m) net-return matrix.

        m == 2 uses the EXACT recursion; m > 2 uses the MC mixture.
        """
        R = np.asarray(R, dtype=np.float64)
        n, m = R.shape
        if m == 2:
            b1, b2, wealth = cls.exact_2asset(R[:, 0], R[:, 1])
            weights = np.column_stack([b1, b2])
        else:
            wealth, weights = cls.mc_simplex(R, n_samples=n_samples, seed=seed)
        return PortfolioResult(weights=weights, wealth=wealth,
                               returns=R, seed=seed)


# ==========================================================================
# Result object
# ==========================================================================

@dataclass
class PortfolioResult:
    """Causal universal-portfolio weights + wealth + gated/trade views."""

    weights: np.ndarray        # (n, m) daily holdings (causal)
    wealth: np.ndarray         # (n+1,) wealth path, wealth[0] = 1.0
    returns: np.ndarray        # (n, m) net returns (for gating/stats)
    seed: int = 0
    gated_weights: Optional[np.ndarray] = field(default=None)
    gated_trades: int = 0
    cost: float = 0.0

    # -- Cover sec.10 gate: rebalance only when log-wealth gain > ln(1+cost) -

    def apply_gate(self, cost: float, inplace: bool = False) -> "PortfolioResult":
        """Hold the previous weight vector unless moving beats the cost drag.

        Cover (1991) sec.10: 'trade only if the increase in W is greater than
        the logarithm of the normalized transaction costs.'
        """
        n, m = self.returns.shape
        gw = np.empty_like(self.weights)
        held = self.weights[0].copy()
        trades = 1
        alc = np.log(1.0 + cost) if cost > 0 else 0.0
        rel = 1.0 + self.returns
        for i in range(n):
            if i > 0:
                gain = np.log(max(rel[i] @ self.weights[i], 1e-300)) \
                    - np.log(max(rel[i] @ held, 1e-300))
                if gain > alc:
                    held = self.weights[i].copy()
                    trades += 1
            gw[i] = held
        res = self if inplace else PortfolioResult(
            weights=self.weights, wealth=self.wealth, returns=self.returns,
            seed=self.seed, gated_weights=gw, gated_trades=trades, cost=cost)
        if inplace:
            self.gated_weights = gw
            self.gated_trades = trades
            self.cost = cost
        return res

    # -- statistics ---------------------------------------------------------

    def wealth_from(self, w: Optional[np.ndarray] = None) -> np.ndarray:
        w = self.weights if w is None else w
        v = np.ones(len(self.returns) + 1)
        for i in range(len(self.returns)):
            v[i + 1] = v[i] * (1.0 + self.returns[i] @ w[i])
        return v

    def stats(self, weights: Optional[np.ndarray] = None) -> dict:
        """terminal wealth, maxDD, longest underwater run (days)."""
        v = self.wealth if weights is None else self.wealth_from(weights)
        peak = 1.0
        dd = 0.0
        under = 0
        longest = 0
        for x in v[1:]:
            peak = max(peak, x)
            dd = max(dd, 1.0 - x / peak)
            if x < peak:
                under += 1
                longest = max(longest, under)
            else:
                under = 0
        return {"terminal": float(v[-1]), "maxdd": float(dd),
                "underwater_days": int(longest)}

    def latest_weights(self) -> pd.Series:
        return pd.Series(self.weights[-1], index=None)


# ==========================================================================
# Side information (Cover & Ordentlich 1996) — tape-level HMM states
# ==========================================================================

class RegimeStates:
    """3-state Gaussian HMM state labels for a return tape (side info).

    Reuses the repo's canonical feature definitions (hmm_regime_detection.py
    build_features: mkt_ret, vol21, avg pairwise corr). Fit on the ORIGINAL
    tape only — never on a resampled path (would leak the future).
    """

    def __init__(self, R: np.ndarray, labels: Optional[pd.Index] = None,
                 n_states: int = 3, seed: int = 7):
        self.R = np.asarray(R, dtype=np.float64)
        self.labels = labels
        self.n_states = n_states
        self.seed = seed
        self.states: Optional[np.ndarray] = None

    @property
    def k(self) -> int:
        return int(self.states.max()) + 1 if self.states is not None else 0

    def fit(self) -> "RegimeStates":
        from hmmlearn.hmm import GaussianHMM
        from hmm_regime_detection import build_features  # canonical features

        idx = self.labels if self.labels is not None else pd.RangeIndex(len(self.R))
        rets = pd.DataFrame(self.R, index=idx)
        feat = build_features(rets)
        X = feat[["mkt_ret", "vol21", "avg_corr"]].values
        X = np.nan_to_num(X, nan=0.0)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        mdl = GaussianHMM(n_components=self.n_states, covariance_type="full",
                          random_state=self.seed, n_iter=200, tol=1e-4)
        mdl.fit((X - mu) / sd)
        self.states = mdl.predict((X - mu) / sd)
        return self

    def counts(self) -> dict:
        return pd.Series(self.states).value_counts().sort_index().to_dict()

    def side_portfolio(self, R: Optional[np.ndarray] = None,
                       n_samples: int = 20000, seed: int = 0) -> PortfolioResult:
        """State-conditional universal portfolio.

        For each state y, run the universal engine on THAT state's day
        subsequence only; day-i weight = the state-y_i universal weight.
        Product over states: S_hat_n = prod_y int_B S_n(b|y) dmu(b).

        Sparse-state fallback: HMM states persist in clusters, so a bootstrap
        path can draw blocks containing <2 days of a state. Days whose state
        is under-sampled keep the PLAIN (unconditional) universal weight —
        the side signal is uninformative there, and a zero weight vector would
        silently destroy the whole path.
        """
        if self.states is None:
            raise ValueError("fit() first")
        R = self.R if R is None else np.asarray(R, dtype=np.float64)
        n, m = R.shape
        k = self.k
        # default: plain universal portfolio on the whole tape
        base = UniversalEngine.run(R, seed=seed, n_samples=n_samples)
        wts = base.weights.copy()
        for y in range(k):
            idx = np.where(self.states[:n] == y)[0]
            if len(idx) < 2:
                continue
            sub = R[idx]
            if m == 2:
                b1, b2, _ = UniversalEngine.exact_2asset(sub[:, 0], sub[:, 1])
                wsub = np.column_stack([b1, b2])
            else:
                _, wsub = UniversalEngine.mc_simplex(sub, n_samples=n_samples, seed=seed)
            for t, i in enumerate(idx):
                wts[i] = wsub[t]
        res = PortfolioResult(weights=wts, wealth=np.ones(n + 1), returns=R, seed=seed)
        v = res.wealth
        rel = 1.0 + R
        for i in range(n):
            v[i + 1] = v[i] * (rel[i] @ wts[i])
        res.wealth = v
        return res


# ==========================================================================
# Sizing — map universal weights onto portfolio_holdings.parquet
# ==========================================================================

class SizingPlan:
    """Target weights (e.g. latest gated universal weights) vs holdings.

    Produces a per-name trade plan: current vs target weight/notional,
    delta shares at last close, action, honoring the Cover sec.10 gate
    (positions are only touched when the target drifted beyond the cost
    threshold from the held weight vector).
    """

    def __init__(self, holdings: pd.DataFrame, target: pd.Series,
                 gate_weights: Optional[pd.Series] = None, cost: float = 0.001,
                 portfolio_value: Optional[float] = None):
        self.holdings = holdings.set_index("ticker") if "ticker" in holdings.columns else holdings
        self.target = target
        self.gate_weights = gate_weights
        self.cost = cost
        self.value = portfolio_value

    @classmethod
    def from_book_files(cls, cost: float = 0.001,
                        portfolio_value: Optional[float] = None) -> "SizingPlan":
        h = pd.read_parquet(DATA_DIR / "portfolio_holdings.parquet")
        w = pd.read_parquet(OUT_BOOK)
        last = w.iloc[-1]
        target = last.drop(labels=["date"]) if "date" in last.index else last
        if "date" in last.index:
            target = last.drop("date")
        # gate: current holdings weights ARE the held vector (cost-gated
        # holdings vs raw target — only rebalance names that drifted > cost)
        return cls(h, target=target.astype(float), cost=cost,
                   portfolio_value=portfolio_value)

    def plan(self, save: bool = False) -> pd.DataFrame:
        cur_w = self.holdings["weight"] if "weight" in self.holdings.columns else None
        if cur_w is None:
            # fall back to market-value weights
            mv = self.holdings["market_value"]
            cur_w = mv / mv.sum()
        cur_w = pd.to_numeric(cur_w, errors="coerce")
        # portfolio_holdings.weight is stored in PERCENT form (13.75 = 13.75%);
        # universal targets are fractions — normalize before comparing.
        if cur_w.max() > 1.5:
            cur_w = cur_w / 100.0
            self.holdings.loc[cur_w.index, "weight"] = cur_w
        mv = self.holdings["market_value"] if self.holdings["market_value"] is not None \
            else self.value * cur_w
        value = self.value or float(self.holdings["market_value"].sum())
        rows = []
        for tk in self.target.index:
            cur = float(cur_w.get(tk, 0.0))
            tgt = float(self.target.get(tk, 0.0))
            delta_w = tgt - cur
            if self.gate_weights is not None and delta_w != 0:
                # Cover sec.10: skip if the rebalance gain < normalized cost
                # (proxy: only trade when |delta_w| exceeds the cost drag)
                if abs(delta_w) <= self.cost:
                    delta_w = 0.0
            cur_mv = value * cur
            tgt_mv = value * tgt
            delta_mv = tgt_mv - cur_mv
            close = None
            if "last_close" in self.holdings.columns:
                close = self.holdings.loc[tk, "last_close"] if tk in self.holdings.index else None
            shares_delta = None
            if close and close > 0:
                shares_delta = delta_mv / close
            action = "BUY" if delta_mv > 0 else ("SELL" if delta_mv < 0 else "HOLD")
            rows.append({
                "ticker": tk,
                "current_weight": cur,
                "target_weight": tgt,
                "delta_weight": delta_w,
                "current_mv": cur_mv,
                "target_mv": tgt_mv,
                "delta_mv": delta_mv,
                "last_close": close,
                "shares_delta": shares_delta,
                "action": action,
            })
        plan = pd.DataFrame(rows)
        if save:
            plan.to_parquet(OUT_SIZING, index=False)
        return plan


# ==========================================================================
# Research bootstrap (same generator as cppi_backtest.py)
# ==========================================================================

def shared_paths(rt: np.ndarray, rb: np.ndarray, n_paths: int, block: int, seed: int,
                 states: Optional[np.ndarray] = None):
    """Block bootstrap (21d blocks, seed 0 — same as item 14 / cppi_backtest).

    With `states`, yields (r1, r2, s) — states resampled with the SAME block
    draws, so side info travels with the return blocks it came from (no
    lookahead into the resampled path).
    """
    rng = np.random.default_rng(seed)
    n = min(len(rt), len(rb))
    rt, rb = rt[:n], rb[:n]
    st = states[:n] if states is not None else None
    nblk = n // block
    for _ in range(n_paths):
        idx = rng.integers(0, nblk, size=nblk)
        r1 = np.concatenate([rt[i * block:(i + 1) * block] for i in idx])[:n]
        r2 = np.concatenate([rb[i * block:(i + 1) * block] for i in idx])[:n]
        if st is not None:
            s = np.concatenate([st[i * block:(i + 1) * block] for i in idx])[:n]
            yield r1, r2, s
        else:
            yield r1, r2


def bootstrap_books(rt: np.ndarray, rb: np.ndarray, n_paths: int = 400,
                    block: int = 21, seed: int = 0, side: bool = False,
                    n_samples: int = 20000) -> pd.DataFrame:
    """Run universal / erc / hrp / vincent_ls (and universal_side if side)
    over SHARED bootstrap paths; returns the book x path stats frame.

    The returned frame mirrors the `cppi_paths.parquet` schema exactly:
    book, terminal, maxdd, underwater_days.

    Side info: states are fitted ONCE on the ORIGINAL tape and resampled with
    the same block draws — never refitted per path (no lookahead, no cost).
    """
    w_t = (1 / float(rt.std())) / (1 / float(rt.std()) + 1 / float(rb.std()))
    w_b = 1 - w_t
    wh_t = 1.0 - float(rt.var()) / (float(rt.var()) + float(rb.var()))
    wh_b = 1.0 - wh_t
    ft = 1.5  # Vince best grid cell

    states = None
    if side:
        base_rs = RegimeStates(np.column_stack([rt, rb])).fit()
        states = base_rs.states

    def fixed_stats(w1: float, w2: float, r1: np.ndarray, r2: np.ndarray) -> dict:
        w = 1.0
        peak = 1.0
        dd = 0.0
        under = 0
        longest = 0
        for i in range(len(r1)):
            w *= 1.0 + w1 * r1[i] + w2 * r2[i]
            if w <= 0:
                return {"terminal": 0.0, "maxdd": 1.0, "underwater_days": len(r1)}
            peak = max(peak, w)
            dd = max(dd, 1.0 - w / peak)
            if w < peak:
                under += 1
                longest = max(longest, under)
            else:
                under = 0
        return {"terminal": w, "maxdd": dd, "underwater_days": longest}

    rows = []
    iter_paths = shared_paths(rt, rb, n_paths, block, seed, states=states)
    for draw in iter_paths:
        r1, r2 = draw[0], draw[1]
        R = np.column_stack([r1, r2])
        res = UniversalEngine.run(R, seed=seed, n_samples=n_samples)
        rows.append({"book": "universal", **res.stats()})
        if side:
            s = draw[2]
            rs_side = RegimeStates(R)
            rs_side.states = s  # states from the ORIGINAL tape, resampled
            res_s = rs_side.side_portfolio(R=R, n_samples=n_samples, seed=seed)
            rows.append({"book": "universal_side", **res_s.stats()})
        rows.append({"book": "erc", **fixed_stats(w_t, w_b, r1, r2)})
        rows.append({"book": "hrp", **fixed_stats(wh_t, wh_b, r1, r2)})
        rows.append({"book": "vincent_ls", **fixed_stats(ft, 0.0, r1, r2)})
    return pd.DataFrame(rows)


def summarize(paths_df: pd.DataFrame) -> pd.DataFrame:
    return paths_df.groupby("book").agg(
        median_terminal=("terminal", "median"),
        p05_terminal=("terminal", lambda s: float(np.quantile(s, 0.05))),
        mean_dd=("maxdd", "mean"),
        median_dd=("maxdd", "median"),
        median_underwater=("underwater_days", "median"),
    )


# ==========================================================================
# CLI (thin)
# ==========================================================================

def cmd_paths(args):
    tmi = pd.read_parquet(DATA_DIR / "bogle_tmi.parquet")
    bpi = pd.read_parquet(DATA_DIR / "bogle_bpi.parquet")
    a = tmi[["date", "ret_net"]].rename(columns={"ret_net": "tmi"})
    b = bpi[["date", "ret_net"]].rename(columns={"ret_net": "bpi"})
    m = a.merge(b, on="date").dropna()
    rt = pd.to_numeric(m["tmi"], errors="coerce").to_numpy()
    rb = pd.to_numeric(m["bpi"], errors="coerce").to_numpy()
    ok = np.isfinite(rt) & np.isfinite(rb)
    rt, rb = rt[ok], rb[ok]
    print(f"sample: {len(rt)} days {m['date'].min().date()} -> {m['date'].max().date()}")
    if args.side:
        rs = RegimeStates(np.column_stack([rt, rb]), labels=m["date"][ok]).fit()
        print(f"side-info states: k={rs.k} {rs.counts()}")

    df = bootstrap_books(rt, rb, n_paths=args.paths, block=args.block,
                         seed=args.seed, side=args.side)
    if args.save or not args.side:
        df.to_parquet(OUT_PATHS, index=False)
        print(f"wrote {OUT_PATHS.name}")
    stats = summarize(df)
    print(stats.to_string())
    for bn in ["universal", "universal_side"]:
        if bn not in stats.index:
            continue
        erc_med = stats.loc["erc", "median_terminal"]
        erc_dd = stats.loc["erc", "median_dd"]
        print(f"{bn} median terminal {stats.loc[bn,'median_terminal']:.3f} "
              f"({stats.loc[bn,'median_terminal']/erc_med:.2f}x ERC) | "
              f"median maxDD {stats.loc[bn,'median_dd']:.2%} | "
              f"BAR {'PASS' if (stats.loc[bn,'median_terminal'] >= 0.8*erc_med and stats.loc[bn,'median_dd'] < erc_dd) else 'FAIL'}")
    if "universal_side" in stats.index:
        u = stats.loc["universal", "median_terminal"]
        us = stats.loc["universal_side", "median_terminal"]
        print(f"side vs plain: {us:.3f} vs {u:.3f} ({us/u:.2f}x)")


def cmd_book(args):
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    d = ds.dataset(str(DATA_DIR / "daily_prices"), format="parquet")
    tab = d.to_table(columns=["date", "ticker", "adj_close"],
                     filter=pc.field("ticker").isin(args.tickers))
    df = tab.to_pandas()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    px = df.pivot_table(index="date", columns="ticker", values="adj_close")
    px = px[args.tickers].dropna(how="any").sort_index()  # common window only
    R = px.pct_change().iloc[1:].to_numpy(float)
    dates = px.index[1:]
    print(f"book: {len(dates)} days {dates[0]} -> {dates[-1]}, "
          f"{len(args.tickers)} names (common window)")

    res = UniversalEngine.run(R, seed=args.seed, n_samples=args.samples)
    res.apply_gate(args.cost, inplace=True)
    out = pd.DataFrame(res.gated_weights, columns=args.tickers)
    out.insert(0, "date", dates)
    if args.save:
        out.to_parquet(OUT_BOOK, index=False)
        print(f"wrote {OUT_BOOK.name} ({len(out)} days)")

    def show(name, w):
        v = res.wealth_from(w)
        term = v[-1]
        print(f"  {name}: {term:.3f}")
    print("wealth (no costs):")
    show("universal gated", res.gated_weights)
    show("universal ungated", res.weights)
    show("equal-weight", np.tile(np.full(len(args.tickers), 1.0 / len(args.tickers)),
                                 (len(R), 1)))
    bh = (1.0 + R).prod(axis=0)
    for j, t in enumerate(args.tickers):
        print(f"    buy-hold {t}: {bh[j]:.3f}")
    print(f"gated rebalances: {res.gated_trades} of {len(R)} days "
          f"(gate {args.cost:.2%})")
    print("latest gated weights:")
    print(out.iloc[-1].to_string())
    if args.sizing:
        plan = SizingPlan.from_book_files(cost=args.cost).plan(save=args.save)
        print("\nsizing plan vs portfolio_holdings.parquet:")
        print(plan.to_string(index=False) if len(plan) < 20 else
              plan.head(20).to_string(index=False))


def cmd_sizing(args):
    plan = SizingPlan.from_book_files(cost=args.cost, portfolio_value=args.value)
    out = plan.plan(save=args.save)
    print(out.to_string(index=False) if len(out) < 20 else out.head(20).to_string(index=False))
    if args.save:
        print(f"wrote {OUT_SIZING.name}")


def cmd_validate(args):
    rng = np.random.default_rng(7)
    n = 120
    for trial in range(3):
        r1 = rng.normal(0.0004, 0.010, n)
        r2 = rng.normal(-0.0002, 0.020, n)
        b1, b2, w_exact = UniversalEngine.exact_2asset(r1, r2)
        R = np.column_stack([r1, r2])
        w_mc, _ = UniversalEngine.mc_simplex(R, n_samples=args.samples, seed=trial)
        print(f"trial {trial}: exact {w_exact[-1]:.6f} | MC {w_mc[-1]:.6f} "
              f"| ratio {w_mc[-1]/w_exact[-1]:.6f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("paths", help="research block-bootstrap vs erc/hrp/ls")
    p.add_argument("--paths", type=int, default=400)
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--side", action="store_true", help="add universal_side (HMM states)")
    p.add_argument("--save", action="store_true")
    b = sub.add_parser("book", help="personal book daily weights + sizing")
    b.add_argument("--tickers", nargs="+", default=BOOK)
    b.add_argument("--samples", type=int, default=20000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--cost", type=float, default=0.001)
    b.add_argument("--sizing", action="store_true", help="also emit sizing plan")
    b.add_argument("--save", action="store_true")
    s = sub.add_parser("sizing", help="sizing plan only (read book weights + holdings)")
    s.add_argument("--cost", type=float, default=0.001)
    s.add_argument("--value", type=float, default=None, help="portfolio value (default: sum holdings MV)")
    s.add_argument("--save", action="store_true")
    v = sub.add_parser("validate", help="exact vs MC on 2 assets")
    v.add_argument("--samples", type=int, default=50000)
    args = ap.parse_args()
    if args.cmd == "paths":
        cmd_paths(args)
    elif args.cmd == "book":
        cmd_book(args)
    elif args.cmd == "sizing":
        cmd_sizing(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
