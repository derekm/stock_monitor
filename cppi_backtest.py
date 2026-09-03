"""Item 14 — Perold/Sharpe: CPPI vs ERC vs HRP vs Vince LS.

Block-bootstrap paths (21-day blocks, same generator as kelly.py ls-vs-erc,
seed 0, 400 paths) over the rebuilt-real TMI/BPI daily returns. Same paths
for every book (kelly.py drew ls and erc on different draws; here the paths
are SHARED so books are compared apples-to-apples).

Books:
  cppi_m2/3/4  CPPI on TMI: floor = 0.9 * peak wealth, exposure = min(m*(w-floor), 1.0),
               remainder in cash (0%). No leverage beyond the cushion cap.
  erc          fixed inverse-vol weights on (TMI, BPI) — same w as kelly.py ERC.
  hrp          fixed HRP weights (portfolio_construction.hrp_weights_from_cov).
  vincent_ls   f=1.50 TMI, 0 BPI (best grid cell, capped exposure at 1.5 — as run in kelly.py).

Bar: CPPI maxDD < ERC maxDD AND median terminal not worse than ERC median by 20%
     (any m in {2,3,4}; report all).

Output: cppi_paths.parquet (book x path terminal wealth + maxDD + max underwater run).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


def shared_paths(rt: np.ndarray, rb: np.ndarray, n_paths: int, block: int, seed: int):
    """Yield (r_tmi, r_bpi) bootstrap path arrays, one draw per path (shared across books)."""
    rng = np.random.default_rng(seed)
    n = min(len(rt), len(rb))
    rt, rb = rt[:n], rb[:n]
    nblk = n // block
    for _ in range(n_paths):
        idx = rng.integers(0, nblk, size=nblk)
        r1 = np.concatenate([rt[i * block:(i + 1) * block] for i in idx])[:n]
        r2 = np.concatenate([rb[i * block:(i + 1) * block] for i in idx])[:n]
        yield r1, r2


def cpppath(r_tmi: np.ndarray, m: float, floor_frac: float = 0.9):
    """CPPI on one path. Returns (terminal, maxDD, longest underwater run in days)."""
    w = peak = 1.0
    peak_nav = 1.0
    dd = 0.0
    under = 0
    longest = 0
    for r in r_tmi:
        floor = floor_frac * peak
        cushion = max(w - floor, 0.0)
        expo = min(m * cushion, 1.0)
        w *= 1.0 + expo * r
        if w <= 0:
            return 0.0, 1.0, len(r_tmi)
        peak = max(peak, w)
        dd = max(dd, 1.0 - w / peak)
        if w < peak_nav:
            under += 1
            longest = max(longest, under)
        else:
            under = 0
            peak_nav = w
    return w, dd, longest


def fixed_path(r: np.ndarray):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=400)
    ap.add_argument("--block", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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

    # fixed weights
    st, sb = float(rt.std()), float(rb.std())
    w_t = (1 / st) / (1 / st + 1 / sb)
    w_b = 1 - w_t
    cov = pd.DataFrame(np.cov(np.vstack([rt, rb])), index=["tmi", "bpi"], columns=["tmi", "bpi"])
    # HRP bisection for n=2 assets reduces analytically to inverse-VARIANCE
    # weights (w1 = 1 - v1/(v1+v2)); portfolio_construction.hrp_weights_from_cov
    # crashes on 2-asset frames (np.ix_ label indexing), so use the closed form.
    v1 = float(cov.loc["tmi", "tmi"])
    v2 = float(cov.loc["bpi", "bpi"])
    wh_t = 1.0 - v1 / (v1 + v2)
    wh_b = 1.0 - wh_t
    ft = 1.5  # Vince best grid cell (f_tmi=1.50, f_bpi=0)
    print(f"ERC w_tmi={w_t:.2f} | HRP w_tmi={wh_t:.2f} | Vince LS f_tmi={ft}")

    rows = []
    for r1, r2 in shared_paths(rt, rb, args.paths, args.block, args.seed):
        for name, fn in [
            ("cppi_m2", lambda: cpppath(r1, 2.0)),
            ("cppi_m3", lambda: cpppath(r1, 3.0)),
            ("cppi_m4", lambda: cpppath(r1, 4.0)),
            ("erc", lambda: fixed_path(w_t * r1 + w_b * r2)),
            ("hrp", lambda: fixed_path(wh_t * r1 + wh_b * r2)),
            ("vincent_ls", lambda: fixed_path(ft * r1)),
        ]:
            term, dd, under = fn()
            rows.append({"book": name, "terminal": term, "maxdd": dd, "underwater_days": under})
    df = pd.DataFrame(rows)
    df.to_parquet(DATA_DIR / "cppi_paths.parquet", index=False)

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
    print(f"\nERC median terminal {erc_med:.3f} | ERC median maxDD {erc_dd:.2%}")
    for m in ["cppi_m2", "cppi_m3", "cppi_m4"]:
        dd_ok = stats.loc[m, "median_dd"] < erc_dd
        tw_ok = stats.loc[m, "median_terminal"] >= 0.8 * erc_med
        print(f"{m}: DD {stats.loc[m,'median_dd']:.2%} < ERC? {dd_ok} | terminal {stats.loc[m,'median_terminal']:.3f} >= 0.8x ERC? {tw_ok} -> {'PASS' if dd_ok and tw_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
