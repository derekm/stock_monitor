"""adhoc_tilt_scan.py — ONE-SHOT: rank candidate tilt buys for the personal
book by the universal engine's OWN marginal improvement, plus
diversification vs the 9 names. Each candidate is added to the book alone
(book+candidate, m=10), same common window, same seed/n_samples; the
marginal terminal ratio vs book-only is the tilt score. This is the honest
"does adding X improve the no-lookahead mix" answer per candidate.
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

# Candidate pool: the 3 gated names + TRV (flagship pair) + universe-screen
# best partner per book name + top buy_candidates by composite.
CANDIDATES = ["ALL", "GL", "EOG", "TRV", "QXO", "SUI", "PHG", "KR", "LINE", "MTN"]

# --- gates (same as the candidate run: veto-clean, implied_r>0 ... but TRV
# may lack implied_r coverage — we gate on veto+fragile+liquid and REPORT
# implied_r separately instead of hard-filtering it here) ---------------
veto = pd.read_parquet(DATA_DIR / "fragility_veto.parquet") \
    if (DATA_DIR / "fragility_veto.parquet").exists() else None
fs = pd.read_parquet(DATA_DIR / "fragility_screen.parquet") \
    if (DATA_DIR / "fragility_screen.parquet").exists() else None
ir = pd.read_parquet(DATA_DIR / "implied_r_screen.parquet") \
    if (DATA_DIR / "implied_r_screen.parquet").exists() else None
liq = liquid_listed_tickers()

def gated(t):
    ok = t in liq
    if veto is not None and len(veto):
        v = veto[veto["ticker"] == t]
        if len(v):
            ok &= not bool(v["veto_flag"].fillna(False).iloc[0] |
                           v["alpha_lt_2"].fillna(False).iloc[0])
    if fs is not None and len(fs):
        f = fs[fs["ticker"] == t]
        if len(f):
            ok &= not bool(f["fragile_flag"].fillna(False).iloc[0])
    return ok

recs = []
for t in CANDIDATES:
    recs.append({"cand": t, "gated": gated(t), "frag": None, "alpha": None, "ir": None})
    if fs is not None and len(fs) and (fs["ticker"] == t).any():
        recs[-1]["frag"] = bool(fs.loc[fs["ticker"] == t, "fragile_flag"].iloc[0])
    if veto is not None and len(veto) and (veto["ticker"] == t).any():
        recs[-1]["alpha"] = round(float(veto.loc[veto["ticker"] == t, "tail_alpha_hill"].iloc[0]), 2)
    if ir is not None and len(ir) and (ir["ticker"] == t).any():
        recs[-1]["ir"] = round(float(ir.loc[ir["ticker"] == t, "implied_r"].iloc[0]), 3)

# --- panel: book + all candidates, common window --------------------------
d = ds.dataset(str(DATA_DIR / "daily_prices"), format="parquet")
tab = d.to_table(columns=["date", "ticker", "adj_close"],
                 filter=pc.field("ticker").isin(BOOK + CANDIDATES))
df = tab.to_pandas()
df["date"] = pd.to_datetime(df["date"]).dt.date
px = df.pivot_table(index="date", columns="ticker", values="adj_close")
px = px[BOOK + CANDIDATES].dropna(how="any").sort_index()
R = px.pct_change().iloc[1:].to_numpy(float)
dates = px.index[1:]
print(f"common window: {len(dates)} days {dates[0]} -> {dates[-1]}")

base = UniversalEngine.run(R[:, [BOOK.index(t) for t in BOOK]], seed=0, n_samples=40000)
base_term = base.stats()["terminal"]
print(f"\nbook-only terminal: {base_term:.3f}")

# avg return correlation of each candidate vs the 9 book names (diversification)
rts = pd.DataFrame(R, columns=BOOK + CANDIDATES, index=dates)

rows = []
for t in CANDIDATES:
    if t not in px.columns:
        continue
    j = BOOK.index(t) if t in BOOK else BOOK + [t]
    cols_idx = [BOOK + CANDIDATES].index if False else None
    idx = [BOOK.index(b) for b in BOOK] + [len(BOOK) + CANDIDATES.index(t)]
    res = UniversalEngine.run(R[:, idx], seed=0, n_samples=40000)
    term = res.stats()["terminal"]
    corr = rts[BOOK].corrwith(rts[t]).mean()
    g = next(r["gated"] for r in recs if r["cand"] == t)
    rows.append({"cand": t, "term": term, "ratio": term / base_term,
                 "pg_improve": (term / base_term - 1) * 100, "avg_corr_book": corr,
                 "gated": g})

out = pd.DataFrame(rows).sort_values("ratio", ascending=False)
print("\n=== per-candidate marginal tilt scan (max improvement = best tilt) ===")
print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

bests = out.head(8)
print("\ncandidates passing gates:", out[out.gated]["cand"].tolist())
best_gated = out[out.gated].head(3)
print("\nrecommendation order (gated, best tilt first):")
for _, r in best_gated.iterrows():
    print(f"  {r['cand']}: +{r['pg_improve']:.1f}% terminal, avg corr vs book {r['avg_corr_book']:.2f}")
