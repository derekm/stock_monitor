"""adhoc_up_candidates.py — ONE-SHOT: extend the 9-name book with gated
candidates and report where the universal portfolio would put cash.

Rule (per docs/RESEARCH_INTEGRATION_PLAN.md item 23 do-nots): the candidate
universe is FIXED BEFORE the UP run, using only external gates — never by
peeking at UP hindsight weights. This script selects candidates from
buy_candidates ∩ implied_r>0 ∩ veto-clean ∩ liquid-listed, and reports the
extended-simplex result as a candidate ORDER, not an instruction.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
import sys

DATA_DIR = Path(__file__).parent
sys.path.insert(0, str(DATA_DIR))
from universal_portfolio import UniversalEngine
from analytics_common import liquid_listed_tickers

BOOK = ["BAYRY", "CAG", "HMC", "HPQ", "KHC", "MOS", "PFE", "SMCI", "T"]
N_CAND = 8           # how many gated candidates to add
CAP = 0.05           # per-candidate size cap (fraction of book)
MIN_OBS = 756        # >= 3y of hive history to enter the common window

# --- 1) fixed candidate universe (external gates only, no UP peeking) -------
bc = pd.read_parquet(DATA_DIR / "buy_candidates.parquet")
ir = pd.read_parquet(DATA_DIR / "implied_r_screen.parquet")
veto = pd.read_parquet(DATA_DIR / "fragility_veto.parquet") \
    if (DATA_DIR / "fragility_veto.parquet").exists() else None
liq = liquid_listed_tickers()

gates = (bc["ticker"].isin(ir.loc[ir["implied_r"] > 0, "ticker"]) &
         bc["ticker"].isin(liq))
if veto is not None and len(veto):
    clean = set(veto.loc[~(veto["veto_flag"].fillna(False) |
                           veto["alpha_lt_2"].fillna(False)), "ticker"])
    gates &= bc["ticker"].isin(clean)
cand = bc.loc[gates & ~bc["ticker"].isin(BOOK)]
cand = cand.sort_values("composite_score", ascending=False) \
    if "composite_score" in cand.columns else cand.sort_values("rank")
cand = cand.head(N_CAND)["ticker"].tolist()
print(f"gated candidates ({len(cand)}): {cand}")

# --- 2) common-window panel over book + candidates --------------------------
d = ds.dataset(str(DATA_DIR / "daily_prices"), format="parquet")
tab = d.to_table(columns=["date", "ticker", "adj_close"],
                 filter=pc.field("ticker").isin(BOOK + cand))
df = tab.to_pandas()
df["date"] = pd.to_datetime(df["date"]).dt.date
px = df.pivot_table(index="date", columns="ticker", values="adj_close")

cov = px.notna().sum()
keep = [t for t in BOOK + cand if cov.get(t, 0) >= MIN_OBS]
print(f"coverage >= {MIN_OBS} obs: {len(keep)}/{len(BOOK) + len(cand)} names")

px = px[keep].dropna(how="any").sort_index()
R = px.pct_change().iloc[1:].to_numpy(float)
dates = px.index[1:]
print(f"common window: {len(dates)} days {dates[0]} -> {dates[-1]}")

book_idx = [i for i, t in enumerate(keep) if t in BOOK]
cand_new = [t for t in keep if t not in BOOK]

# --- 3) run UP on book-only (control) and book+candidates (extended) --------
res_base = UniversalEngine.run(R[:, book_idx], seed=0, n_samples=60000)
res_ext = UniversalEngine.run(R, seed=0, n_samples=60000)

w_end_ext = res_ext.weights[-1]
w_252 = res_ext.weights[-252:].mean(axis=0) if len(R) >= 252 else res_ext.weights.mean(axis=0)

print("\n=== extended-simplex result (candidate ORDER, not instruction) ===")
print(f"{'name':6} {'latest w':>9} {'mean 252d':>10} {'stable':>7}")
rows = []
for j, t in enumerate(keep):
    if t in BOOK:
        continue
    latest = float(w_end_ext[j])
    mean252 = float(w_252[j])
    stable = "yes" if (latest > 0 and mean252 > 0 and
                       abs(latest - mean252) / max(mean252, 1e-9) < 0.5) else "no"
    rows.append((t, latest, mean252, stable))
for t, latest, mean252, stable in sorted(rows, key=lambda r: -r[1]):
    print(f"{t:6} {latest:9.1%} {mean252:10.1%} {stable:>7}")

ext_term = res_ext.stats()["terminal"]
base_term = res_base.stats()["terminal"]
print(f"\nterminal on common window: book-only {base_term:.3f} | "
      f"+candidates {ext_term:.3f} ({ext_term/base_term:.2f}x)")

LAST_CLOSE = px.iloc[-1]
print("\n=== dollar sizes at 5% cap (today's closes) ===")
# portfolio value from holdings
h = pd.read_parquet(DATA_DIR / "portfolio_holdings.parquet")
value = float(h["market_value"].sum()) if "market_value" in h.columns else 307.0
funded = 0.0
for t, latest, mean252, stable in sorted(rows, key=lambda r: -r[1]):
    if latest <= 0:
        continue
    sh = CAP * value / float(LAST_CLOSE[t])
    print(f"  {t:6} ${CAP*value:6.2f} max ({sh:.2f} sh @ {float(LAST_CLOSE[t]):.2f}) "
          f"{'— stable weight' if stable else '— UNSTABLE, verify'}")
    funded += CAP * value
print(f"\nmax total cash deployed to candidates: ${funded:.2f} ({funded/value:.0%} of book)")

# bought-kept book names that would be trimmed to fund it (top shifts)
w0, w1 = res_base.weights[-1], res_ext.weights[-1]
print("\nbook weight shift (extended vs book-only):")
for j, t in enumerate(keep):
    if t in BOOK:
        i = book_idx[j]
        print(f"  {t:6} {w0[i]:7.1%} -> {w1[j]:7.1%} ({(w1[j]-w0[i])*100:+.1f}pp)")
