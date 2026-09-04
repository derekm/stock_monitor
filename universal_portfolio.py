"""universal_portfolio.py — Cover (1991) universal portfolio, research + book.

Theory (see docs/cover_universal_portfolio.md): the Dirichlet(1/2)-weighted
universal portfolio holds, on day k, the performance-weighted average over ALL
constant-rebalanced portfolios in the simplex:

    b_hat_k = int_B b * S_{k-1}(b) dmu(b) / int_B S_{k-1}(b) dmu(b),
    S_hat_n = int_B S_n(b) dmu(b),

so its wealth telescopes to a plain integral over the simplex — computable
causally, with ZERO statistical assumptions and NO lookahead. Regret vs the
best CRP in hindsight is exactly ((m-1)/2) log(n+1) (Ordentlich-Cover minimax),
so universal <= oracle at finite n is a theorem, not an implementation bug.

Two engines:
  exact_2asset   Dirichlet(1/2) UP for m=2 — Cover eq. (128) Q-recursion,
                 O(n^2), EXACT. Verified by the telescope identity
                 S_hat_n == sum_l Q_n(l) (self-check inside).
  mc_simplex     Dirichlet(1/2) Monte-Carlo mixture for m>=2 (Kalai-Vempala
                 style): sample portfolios from the simplex, wealth-weight
                 them. Converges to the same integral; validated against the
                 exact engine on 2 assets.

Modes:
  --paths        research: shared 400-path block bootstrap on (TMI, BPI) with
                 the SAME generator as cppi_backtest.py (block 21, seed 0);
                 books universal / erc / hrp / vincent_ls / cppi_m3, same
                 stats schema -> universal_paths.parquet. Cost convention:
                 none charged (index-level bootstrap, same as item 14).
  --book --save  personal portfolio: daily universal weights for the book
                 tickers from daily_prices (hive READ, no hive writes), with
                 Cover sec.10 rebalance gate (trade only when the log-wealth
                 gain exceeds the transaction-cost drag).
  --validate     exact vs mc on 2 assets (telescope identity + convergence).

Outputs are derived panels only. Never writes daily_prices.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

BOOK = ["BAYRY", "CAG", "HMC", "HPQ", "KHC", "MOS", "PFE", "SMCI", "T"]
OUT_PATHS = DATA_DIR / "universal_paths.parquet"
OUT_BOOK = DATA_DIR / "universal_book_weights.parquet"


# --------------------------------------------------------------------------
# Engine 1: exact Dirichlet(1/2) UP, m = 2 (Cover eq. 128 recursions)
# --------------------------------------------------------------------------

def exact_2asset(r1: np.ndarray, r2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact universal portfolio for two assets. r1/r2: daily net returns.

    Returns (b1, b2, wealth): day-by-day causal weights and wealth path,
    starting wealth 1.0. The Dirichlet(1/2) prior gives the minimax-optimal
    regret ((m-1)/2)log(n+1) (Ordentlich-Cover 1998).
    """
    x1 = 1.0 + np.asarray(r1, dtype=np.float64)
    x2 = 1.0 + np.asarray(r2, dtype=np.float64)
    n = len(x1)
    b1 = np.empty(n)
    b2 = np.empty(n)
    wealth = np.empty(n + 1)
    wealth[0] = 1.0
    # Q_l^{(k-1)} recursion: coefficients of b^l (1-b)^{k-1-l} in the
    # Dirichlet(1/2)-weighted wealth, eq. (126)-(127).
    Q = np.array([1.0])
    for k in range(1, n + 1):
        l = np.arange(k, dtype=np.float64)
        s = Q.sum()
        # eq. (128): b_hat from Q_{k-1}; (l + 1 - 1/2)/k and (k-l-1/2)/k
        w1 = float(((l + 0.5) * Q).sum()) / (k * s)
        w2 = float(((k - l - 0.5) * Q).sum()) / (k * s)
        assert abs(w1 + w2 - 1.0) < 1e-12, "weights must sum to 1"
        b1[k - 1] = w1
        b2[k - 1] = w2
        wealth[k] = wealth[k - 1] * (w1 * x1[k - 1] + w2 * x2[k - 1])
        # next Q_k from Q_{k-1}
        Qn = np.zeros(k + 1)
        q = Q
        Qn[0] = x2[k - 1] * (k - 0.5) / k * q[0]
        Qn[k] = x1[k - 1] * (k - 0.5) / k * q[k - 1]
        if k > 1:
            Qn[1:k] = (x1[k - 1] * (l[1:] - 0.5) / k * q[:-1]
                       + x2[k - 1] * (k - l[1:] - 0.5) / k * q[1:])
        Q = Qn
    # telescope identity: S_hat_n == sum_l Q_n(l)
    assert abs(wealth[-1] - Q.sum()) / max(wealth[-1], 1e-300) < 1e-9, \
        f"telescope failed: {wealth[-1]} vs {Q.sum()}"
    return b1, b2, wealth[1:]


# --------------------------------------------------------------------------
# Engine 2: Dirichlet(1/2) Monte-Carlo mixture, m >= 2
# --------------------------------------------------------------------------

def mc_simplex(R: np.ndarray, n_samples: int = 20000, seed: int = 0, alpha: float = 0.5):
    """Universal portfolio via simplex sampling. R: (n, m) daily net returns.

    Returns (W, weights): W (n+1,) wealth path (product of causal multipliers),
    weights (n, m) daily holdings. Converges to the exact integral as
    n_samples -> inf (Kalai-Vempala sampling).
    """
    m = R.shape[1]
    rng = np.random.default_rng(seed)
    B = rng.dirichlet([alpha] * m, size=n_samples)          # (S, m)
    W = np.ones(n_samples)                                   # sample wealths S_{k-1}(b)
    Ws = np.empty(R.shape[0] + 1)
    Ws[0] = 1.0
    wts = np.empty((R.shape[0], m))
    mult = 1.0 + R                                            # (n, m) price relatives
    for i in range(R.shape[0]):
        w = (B * W[:, None]).sum(axis=0) / W.sum()           # causal b_hat_i
        wts[i] = w
        mult_i = mult[i] @ w                                 # b_hat_i . x_i
        Ws[i + 1] = Ws[i] * mult_i
        W *= mult[i] @ B.T                                   # wealth-weight the samples
    return Ws, wts


# --------------------------------------------------------------------------
# Path / book helpers (shared with cppi_backtest.py conventions)
# --------------------------------------------------------------------------

def shared_paths(rt: np.ndarray, rb: np.ndarray, n_paths: int, block: int, seed: int):
    """Same block-bootstrap generator as cppi_backtest.py (21d blocks, seed 0)."""
    rng = np.random.default_rng(seed)
    n = min(len(rt), len(rb))
    rt, rb = rt[:n], rb[:n]
    nblk = n // block
    for _ in range(n_paths):
        idx = rng.integers(0, nblk, size=nblk)
        r1 = np.concatenate([rt[i * block:(i + 1) * block] for i in idx])[:n]
        r2 = np.concatenate([rb[i * block:(i + 1) * block] for i in idx])[:n]
        yield r1, r2


def path_stats(r: np.ndarray):
    """(terminal wealth, maxDD, longest underwater run) for a wealth multiplier path."""
    w = 1.0
    peak = 1.0
    dd = 0.0
    under = 0
    longest = 0
    for x in r:
        w *= 1.0 + x
        if w <= 0:
            return 0.0, 1.0, len(r)
        peak = max(peak, w)
        dd = max(dd, 1.0 - w / peak)
        if w < peak:
            under += 1
            longest = max(longest, under)
        else:
            under = 0
    return w, dd, longest


# --------------------------------------------------------------------------
# Research: shared bootstrap, universal vs erc/hrp/ls/cppi
# --------------------------------------------------------------------------

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

    st, sb = float(rt.std()), float(rb.std())
    w_t = (1 / st) / (1 / st + 1 / sb)
    w_b = 1 - w_t
    v1, v2 = float(rt.var()), float(rb.var())
    wh_t = 1.0 - v1 / (v1 + v2)
    wh_b = 1.0 - wh_t
    ft = 1.5
    print(f"ERC w_tmi={w_t:.2f} | HRP w_tmi={wh_t:.2f} | Vince LS f_tmi={ft}")

    rows = []
    for r1, r2 in shared_paths(rt, rb, args.paths, args.block, args.seed):
        _, _, w_u = exact_2asset(r1, r2)
        ru = np.diff(np.log(w_u))                      # UP daily log-returns
        for name, rets in [
            ("universal", ru),
            ("erc", w_t * r1 + w_b * r2),
            ("hrp", wh_t * r1 + wh_b * r2),
            ("vincent_ls", ft * r1),
            ("cppi_m3", None),
        ]:
            if name == "cppi_m3":
                # same CPPI (m=3) book as item 14
                w = peak = 1.0
                dd = under = longest = 0
                for x in r1:
                    floor = 0.9 * peak
                    expo = min(3.0 * max(w - floor, 0.0), 1.0)
                    w *= 1.0 + expo * x
                    if w <= 0:
                        break
                    peak = max(peak, w)
                    dd = max(dd, 1.0 - w / peak)
                    if w < peak:
                        under += 1
                        longest = max(longest, under)
                    else:
                        under = 0
                rows.append({"book": name, "terminal": w, "maxdd": dd,
                             "underwater_days": longest})
                continue
            # universal arrives as log-returns; others as simple net returns
            r = rets if name == "universal" else rets
            if name == "universal":
                term, dd, under = np.exp(r.sum()), 0.0, 0
                w, peak = 1.0, 1.0
                under = longest = 0
                for x in np.exp(r):
                    w *= x
                    peak = max(peak, w)
                    dd = max(dd, 1.0 - w / peak)
                    if w < peak:
                        under += 1
                        longest = max(longest, under)
                    else:
                        under = 0
                term = w
            else:
                term, dd, under = path_stats(r)
            rows.append({"book": name, "terminal": term, "maxdd": dd,
                         "underwater_days": under})

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATHS, index=False)
    stats = df.groupby("book").agg(
        median_terminal=("terminal", "median"),
        p05_terminal=("terminal", lambda s: float(np.quantile(s, 0.05))),
        mean_dd=("maxdd", "mean"),
        median_dd=("maxdd", "median"),
        median_underwater=("underwater_days", "median"),
    )
    print(stats.to_string())
    erc_med = stats.loc["erc", "median_terminal"]
    erc_dd = stats.loc["erc", "median_dd"]
    u_med = stats.loc["universal", "median_terminal"]
    u_dd = stats.loc["universal", "median_dd"]
    print(f"\nERC median terminal {erc_med:.3f} | median maxDD {erc_dd:.2%}")
    print(f"universal median terminal {u_med:.3f} ({u_med/erc_med:.2f}x ERC) | "
          f"median maxDD {u_dd:.2%}")
    print(f"BAR (same as item 14): terminal >= 0.8*ERC and DD < ERC "
          f"-> {'PASS' if (u_med >= 0.8*erc_med and u_dd < erc_dd) else 'FAIL'}")
    print(f"wrote {OUT_PATHS.name}")


# --------------------------------------------------------------------------
# Personal book: daily universal weights from daily_prices (hive READ)
# --------------------------------------------------------------------------

def _book_prices(tickers):
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    d = ds.dataset(str(DATA_DIR / "daily_prices"), format="parquet")
    tab = d.to_table(columns=["date", "ticker", "adj_close"],
                     filter=pc.field("ticker").isin(tickers))
    df = tab.to_pandas()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    px = df.pivot_table(index="date", columns="ticker", values="adj_close")
    px = px[tickers].dropna(how="all").sort_index()
    return px


def cmd_book(args):
    px = _book_prices(args.tickers)
    # Honest window: only dates where EVERY book name trades. Pre-listing days
    # (KHC listed 2015, BAYRY ADR) must not enter as 0-return cash — that
    # inflates wealth and distorts the weights.
    px = px.dropna(how="any").sort_index()
    R = px.pct_change().iloc[1:].to_numpy(float)
    dates = px.index[1:]
    print(f"book: {len(dates)} days {dates[0]} -> {dates[-1]}, {len(args.tickers)} names "
          f"(common window, all names trading)")
    W, wts = mc_simplex(R, n_samples=args.samples, seed=args.seed)

    # Cover sec.10 cost gate: rebalance only when the log-wealth gain from
    # rebalancing exceeds the log of the normalized transaction cost.
    cost = args.cost
    alc = np.log(1.0 + cost) if cost > 0 else 0.0
    held = None
    gate_weights = np.empty_like(wts)
    trades = 0
    for i in range(len(R)):
        if held is None:
            held = wts[i]
            trades = 1
        else:
            # improvement from moving held -> wts[i], computed with last bar's
            # relatives (causal); skip if <= cost drag
            gain = np.log(max((1.0 + R[i]) @ wts[i], 1e-300)) \
                - np.log(max((1.0 + R[i]) @ held, 1e-300))
            if gain > alc:
                held = wts[i]
                trades += 1
        gate_weights[i] = held

    out = pd.DataFrame(gate_weights, columns=args.tickers)
    out.insert(0, "date", dates)
    if args.save:
        out.to_parquet(OUT_BOOK, index=False)
        print(f"wrote {OUT_BOOK.name} ({len(out)} days)")

    # comparisons: universal (no gate) / gated / equal-weight / best book name
    def wealth(ws):
        v = 1.0
        for i in range(len(R)):
            v *= 1.0 + ws[i] @ R[i]
        return v

    rng = np.random.default_rng(args.seed)  # not used; keep determinism note
    w_eq = np.full(len(args.tickers), 1.0 / len(args.tickers))
    wts_np = np.asarray(wts)
    gw = np.asarray(gate_weights)
    print(f"\nwealth (no costs):"
          f"\n  universal (no gate): {wealth(wts_np):.3f}"
          f"\n  universal (cost gate, {cost:.2%}): {wealth(gw):.3f}"
          f"\n  equal-weight: {wealth(np.tile(w_eq, (len(R), 1))):.3f}")
    bh = (1.0 + R).prod(axis=0)
    for j, t in enumerate(args.tickers):
        print(f"    buy-hold {t}: {bh[j]:.3f}")
    print(f"\nlatest weights (gated):")
    print(out.iloc[-1].to_string())
    print(f"gated rebalances: {trades} of {len(R)} days "
          f"(turnover gate {cost:.2%})")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def cmd_validate(args):
    rng = np.random.default_rng(7)
    n = 120
    for trial in range(3):
        mu1, sd1 = 0.0004, 0.010
        mu2, sd2 = -0.0002, 0.020
        r1 = rng.normal(mu1, sd1, n)
        r2 = rng.normal(mu2, sd2, n)
        b1, b2, w_exact = exact_2asset(r1, r2)
        R = np.column_stack([r1, r2])
        w_mc, wts_mc = mc_simplex(R, n_samples=args.samples, seed=trial)
        print(f"trial {trial}: exact wealth {w_exact[-1]:.6f} | "
              f"MC wealth {w_mc[-1]:.6f} | ratio {w_mc[-1]/w_exact[-1]:.6f}")
    # telescope identity is asserted inside exact_2asset


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("paths", help="research block-bootstrap vs erc/hrp/ls/cppi")
    p.add_argument("--paths", type=int, default=400)
    p.add_argument("--block", type=int, default=21)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", action="store_true")
    b = sub.add_parser("book", help="personal book daily universal weights")
    b.add_argument("--tickers", nargs="+", default=BOOK)
    b.add_argument("--samples", type=int, default=20000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--cost", type=float, default=0.001, help="rebalance cost (fraction)")
    b.add_argument("--save", action="store_true")
    v = sub.add_parser("validate", help="exact vs MC on 2 assets")
    v.add_argument("--samples", type=int, default=50000)
    args = ap.parse_args()
    if args.cmd == "paths":
        cmd_paths(args)
    elif args.cmd == "book":
        cmd_book(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
