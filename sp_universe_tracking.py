"""
sp_universe_tracking.py — track ALL S&P 500 constituents (503) by index,
basket, and vertical, with our scored inclusion tiers where fundamentals exist.

This fulfills "track all indexes and their baskets": every current constituent
is represented with its GICS sector/sub-industry (the basket/vertical) and, for
the names we hold fundamentals for, our dual_strong / dual_weak / quality_only /
value_only / neither tier + strength_score. Names without fundamentals are
marked metrics_coverage='unscored' (honest — NOT fabricated).

Reads:
  sp500_constituents.parquet  (503, real GICS verticals + date_added)
  fundamentals.parquet        (142, real quarterly PIT fundamentals 2024-2026)
  sp_index_methodology.evaluate  (canonical tiers + S&P-style prediction)

Writes:
  sp500_universe_tracking.parquet  (one row per constituent, current view)
"""
from __future__ import annotations
import datetime as dt
from pathlib import Path

import duckdb

import sp_index_methodology as sim

DATA_DIR = Path(__file__).parent
OUT = DATA_DIR / "sp500_universe_tracking.parquet"


def build(as_of: str | dt.date | None = None) -> list[dict]:
    """Return one tracking row per current constituent."""
    if as_of is None:
        as_of = dt.date.today()
    # our scored tiers for the names we have fundamentals for
    scored = {r["ticker"]: r for r in sim.evaluate(as_of)}

    con = duckdb.connect()
    const = con.execute(
        f"""
        SELECT ticker, name, gics_sector, gics_sub_industry, date_added
        FROM read_parquet('{(DATA_DIR / 'sp500_constituents.parquet').as_posix()}')
        WHERE current
        """
    ).fetchall()

    rows = []
    for tk, name, sector, sub, added in const:
        s = scored.get(tk)
        rows.append({
            "ticker": tk,
            "name": name,
            "vertical": sector,            # GICS sector = basket/vertical
            "sub_vertical": sub,            # GICS sub-industry
            "date_added": added,
            "metrics_coverage": "scored" if s else "unscored",
            "tier": s["tier"] if s else "unscored",
            "strength_score": s["strength_score"] if s else None,
            "buffett_pass": s["buffett_pass"] if s else None,
            "trifecta_pass": s["trifecta_pass"] if s else None,
            "leverage_strong": s["leverage_strong"] if s else None,
            "sp_predicted_member": s["sp_predicted_member"] if s else None,
            "roe": s["roe"] if s else None,
            "roic": s["roic"] if s else None,
            "debt_to_equity": s["debt_to_equity"] if s else None,
            "interest_coverage": s["interest_coverage"] if s else None,
            "ev_ebitda": s["ev_ebitda"] if s else None,
            "pb": s["pb"] if s else None,
            "mktcap_to_assets": s["mktcap_to_assets"] if s else None,
        })
    return rows


def coverage_by_vertical(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    agg = defaultdict(lambda: {"total": 0, "scored": 0, "dual_strong": 0, "dual_weak": 0})
    for r in rows:
        v = r["vertical"] or "Unknown"
        agg[v]["total"] += 1
        if r["metrics_coverage"] == "scored":
            agg[v]["scored"] += 1
        if r["tier"] == "dual_strong":
            agg[v]["dual_strong"] += 1
        if r["tier"] == "dual_weak":
            agg[v]["dual_weak"] += 1
    out = []
    for v, d in agg.items():
        out.append({"vertical": v, **d})
    out.sort(key=lambda x: -x["total"])
    return out


def main():
    rows = build()
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE track (
            ticker VARCHAR, name VARCHAR, vertical VARCHAR, sub_vertical VARCHAR,
            date_added DATE, metrics_coverage VARCHAR, tier VARCHAR,
            strength_score DOUBLE, buffett_pass BOOLEAN, trifecta_pass BOOLEAN,
            leverage_strong BOOLEAN, sp_predicted_member BOOLEAN,
            roe DOUBLE, roic DOUBLE, debt_to_equity DOUBLE, interest_coverage DOUBLE,
            ev_ebitda DOUBLE, pb DOUBLE, mktcap_to_assets DOUBLE
        )
        """
    )
    con.executemany(
        "INSERT INTO track VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (r["ticker"], r["name"], r["vertical"], r["sub_vertical"], r["date_added"],
             r["metrics_coverage"], r["tier"], r["strength_score"], r["buffett_pass"],
             r["trifecta_pass"], r["leverage_strong"], r["sp_predicted_member"],
             r["roe"], r["roic"], r["debt_to_equity"], r["interest_coverage"],
             r["ev_ebitda"], r["pb"], r["mktcap_to_assets"])
            for r in rows
        ],
    )
    con.execute(f"COPY (SELECT * FROM track ORDER BY vertical, ticker) TO '{OUT.as_posix()}' (FORMAT PARQUET)")

    print(f"wrote {OUT} ({len(rows)} constituents)")
    cov = coverage_by_vertical(rows)
    print(f"\n{'vertical':28s} {'tot':>4} {'scored':>6} {'d_str':>5} {'d_weak':>6}")
    for c in cov:
        print(f"{c['vertical']:28s} {c['total']:4d} {c['scored']:6d} "
              f"{c['dual_strong']:5d} {c['dual_weak']:6d}")
    n_scored = sum(1 for r in rows if r["metrics_coverage"] == "scored")
    n_ds = sum(1 for r in rows if r["tier"] == "dual_strong")
    n_dw = sum(1 for r in rows if r["tier"] == "dual_weak")
    print(f"\ncoverage: {n_scored}/{len(rows)} scored; dual_strong={n_ds} dual_weak={n_dw}")
    print("NOTE: 361 constituents lack fundamentals in our store -> 'unscored'.")
    print("      Their vertical/basket tracking is real (GICS); metric fill needs a")
    print("      real fundamentals fetch (not fabricated).")


if __name__ == "__main__":
    main()
