"""
sp_index_methodology.py — S&P 500 inclusion/exclusion reimplementation + our
dual-pass strength tiers, tracked against S&P actuals.

This is the analytics half of the stockmagic <-> stock_monitor integration:

1. S&P METHODOLOGY (reimplemented independently)
   We re-derive what the S&P Index Committee's *published* eligibility factors
   would score for each name, from data we actually have:
     - marketability / size      <- market_cap (liquidity & scale screen)
     - profitability             <- roe > 0 and positive trailing earnings
     - sector representation     <- sp500_sector (committee balances sectors)
     - adequate float            <- unknown from our store (committee factor;
                                   flagged, not scored — documented as a gap)
     - committee discretion      <- explicitly unobservable; we report the
                                   quantifiable score and a predicted flag, then
                                   TRACK it against the real `sp500_member`.

2. OUR INCLUSION / EXCLUSION STRENGTH TIERS (dual strong / dual weak + base)
   Building on the canonical Buffett + trifecta gate (quality_gate_bridge):
     - dual_strong : quality AND value AND strong balance sheet
                     (leverage_strong: D/E <= 1.0 AND interest_coverage >= 3.0)
     - dual_weak   : quality AND value AND acceptable leverage only
                     (D/E <= 2.0 AND interest_coverage >= 1.5) but NOT strong
     - quality_only / value_only / neither : from the base gate
   Plus a continuous `strength_score` (0..1) so conviction can be tracked over
   time, not just bucketed.

3. ACTUALS vs REIMPLEMENTATION
   `compare_to_actuals()` returns precision/recall/agreement of our S&P-style
   prediction against the real `sp500_member` flag — the feedback loop the user
   asked to maintain as our metrics evolve.

Design: duckdb-only at import (pandas optional). Reads stock_monitor's
fundamentals.parquet (PIT) + monitored_stocks.parquet (S&P actuals + sectors).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
_ROOT = DATA_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics import quality_value as qv  # TRIFECTA + canonical thresholds
from src.data import pit_snapshots as pits
from src.data.market_data import MarketDataStore


# --- strength tier thresholds (leverage is the distinguishing axis) ----------
LEV_STRONG_DE_MAX = 1.0
LEV_STRONG_IC_MIN = 3.0
LEV_OK_DE_MAX = 2.0
LEV_OK_IC_MIN = 1.5


def _store() -> MarketDataStore:
    """Register PIT fundamentals + monitored stocks (with S&P actuals)."""
    store = MarketDataStore(":memory:")
    con = store.conn()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE pit_fundamentals AS
        SELECT ticker, as_of_date AS as_of,
               roe, roic, ev_ebitda,
               pb_ratio AS pb, mktcap_to_assets AS mcap_assets,
               debt_to_equity AS debt_equity, interest_coverage
        FROM read_parquet('{(DATA_DIR / 'fundamentals.parquet').as_posix()}')
        """
    )
    # minimal daily_prices axis for build_snapshot_timeseries
    con.execute(
        f"""
        CREATE OR REPLACE TABLE daily_prices AS
        SELECT DISTINCT ticker, as_of_date AS trade_date, NULL AS adj_close
        FROM read_parquet('{(DATA_DIR / 'fundamentals.parquet').as_posix()}')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE monitored_stocks AS
        SELECT m.ticker, m.sector, m.growth_tech_index,
               COALESCE(s.current, FALSE) AS sp500_member, s.gics_sector AS sp500_sector,
               s.date_added AS sp500_date_added, f.market_cap
        FROM read_parquet('{(DATA_DIR / 'monitored_stocks.parquet').as_posix()}') m
        FULL OUTER JOIN read_parquet('{(DATA_DIR / 'sp500_constituents.parquet').as_posix()}') s
          ON s.ticker = m.ticker
        LEFT JOIN (
            SELECT ticker, market_cap FROM read_parquet('{(DATA_DIR / 'fundamentals.parquet').as_posix()}')
            QUALIFY row_number() OVER (PARTITION BY ticker ORDER BY as_of_date DESC) = 1
        ) f ON f.ticker = m.ticker
        """
    )
    return store


def _strength_score(r: dict) -> float:
    """Continuous 0..1 conviction score from the quality/value/leverage legs.

    Each leg contributes proportionally to how far inside its comfortable band
    the metric is; NULL legs contribute 0 (unknown). Weighted toward the
    quality+value core, with a leverage modifier."""
    def clamp(x, lo, hi):
        return max(0.0, min(1.0, (x - lo) / (hi - lo))) if hi > lo else 0.0

    s = 0.0
    w = 0.0
    # quality (Buffett)
    if r.get("roe") is not None:
        s += 0.20 * clamp(r["roe"], 0.10, 0.25); w += 0.20
    if r.get("roic") is not None:
        s += 0.20 * clamp(r["roic"], 0.08, 0.20); w += 0.20
    # value (trifecta)
    if r.get("ev_ebitda") is not None:
        s += 0.15 * clamp(r["ev_ebitda"], 12.0, 4.0); w += 0.15  # lower is better
    if r.get("pb") is not None:
        s += 0.10 * clamp(r["pb"], 2.5, 1.0); w += 0.10
    if r.get("mcap_assets") is not None:
        s += 0.10 * clamp(r["mcap_assets"], 1.0, 0.3); w += 0.10
    # leverage (modifier)
    if r.get("debt_equity") is not None:
        s += 0.15 * clamp(r["debt_equity"], 2.5, 0.3); w += 0.15
    if r.get("interest_coverage") is not None:
        s += 0.10 * clamp(r["interest_coverage"], 1.0, 6.0); w += 0.10
    return round(s / w, 4) if w else 0.0


def evaluate(as_of: str | dt.date | None = None) -> list[dict]:
    """Score every monitored ticker with S&P-style eligibility + our dual
    strength tiers. Returns a list of row dicts."""
    if as_of is None:
        import duckdb
        as_of = duckdb.connect().execute(
            f"SELECT MAX(as_of_date) FROM read_parquet('{(DATA_DIR / 'fundamentals.parquet').as_posix()}')"
        ).fetchone()[0]
    if isinstance(as_of, str):
        as_of = dt.date.fromisoformat(as_of)

    store = _store()
    con = store.conn()
    pits.build_snapshot_timeseries(store, recompute_marketcap=False)
    pits.qualify_as_of(store, as_of)
    qp = con.execute("SELECT * FROM quality_pass").fetchall()
    cols = [d[0] for d in con.description]
    mon = {}
    for rec in con.execute(
        "SELECT ticker, sp500_member, sp500_sector, market_cap FROM monitored_stocks"
    ).fetchall():
        mon[rec[0]] = {"sp500_member": bool(rec[1]), "sp_sector": rec[2], "mcap": rec[3]}

    rows = []
    for rec in qp:
        r = dict(zip(cols, rec))
        tk = r["ticker"]
        buffett = bool(r["buffett_ok"]) and bool(r["leverage_ok"])
        trif = bool(r["trifecta_ok"])
        de = r.get("debt_equity")
        ic = r.get("interest_coverage")
        lev_strong = (de is not None and de <= LEV_STRONG_DE_MAX) and \
                     (ic is not None and ic >= LEV_STRONG_IC_MIN)
        lev_ok = buffett  # leverage_ok already part of buffett gate

        if buffett and trif and lev_strong:
            tier = "dual_strong"
        elif buffett and trif:
            tier = "dual_weak"
        elif buffett:
            tier = "quality_only"
        elif trif:
            tier = "value_only"
        else:
            tier = "neither"

        # --- S&P-style reimplementation (quantifiable factors only) ---
        m = mon.get(tk, {})
        mcap = m.get("mcap")
        sp_sector = m.get("sp_sector")
        roe = r.get("roe")
        sp_eligible = (
            (mcap is not None and mcap >= 1.0e10)   # ~$10B scale/liquidity proxy
            and (roe is not None and roe > 0)        # profitability screen
            and (sp_sector is not None)              # representable sector
        )
        actual = bool(m.get("sp500_member", False))

        rows.append({
            "ticker": tk,
            "tier": tier,
            "strength_score": _strength_score(r),
            "buffett_pass": buffett,
            "trifecta_pass": trif,
            "leverage_strong": lev_strong,
            "sp_predicted_member": sp_eligible,
            "sp_actual_member": actual,
            "sp_sector": sp_sector,
            "roe": r.get("roe"), "roic": r.get("roic"),
            "debt_to_equity": de, "interest_coverage": ic,
            "ev_ebitda": r.get("ev_ebitda"), "pb": r.get("pb"),
            "mktcap_to_assets": r.get("mcap_assets"),
        })
    return rows


def compare_to_actuals(as_of: str | dt.date | None = None) -> dict:
    """Track our S&P-style prediction vs real sp500_member.

    Returns precision / recall / agreement / n, plus tier histograms, so the
    feedback loop is auditable as our methodology evolves.

    n includes the FULL actuals universe (all current constituents), so names
    we lack fundamentals for count as predicted-False / actual-True (a false
    negative for our model, not a data error).
    """
    rows = evaluate(as_of)
    pred = [r for r in rows if r["sp_predicted_member"]]
    actual_rows = {r["ticker"] for r in rows if r["sp_actual_member"]}
    # full actuals universe (every current constituent, even without fundamentals)
    import duckdb
    full_actual = set(r[0] for r in duckdb.connect().execute(
        f"SELECT ticker FROM read_parquet('{(DATA_DIR / 'sp500_constituents.parquet').as_posix()}') "
        f"WHERE current"
    ).fetchall())

    tp = sum(1 for r in rows if r["sp_predicted_member"] and r["sp_actual_member"])
    fp = len(pred) - tp
    fn = len(full_actual - {r["ticker"] for r in pred})
    n = len(full_actual)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    agreement = tp / n if n else None
    from collections import Counter
    tiers = Counter(r["tier"] for r in rows)
    return {
        "as_of": str(as_of) if as_of else "latest",
        "n_actuals_universe": n,
        "n_evaluated": len(rows),
        "sp_predicted": len(pred),
        "sp_actual": len(full_actual),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "agreement": agreement,
        "tier_histogram": dict(tiers),
        "dual_strong": tiers.get("dual_strong", 0),
        "dual_weak": tiers.get("dual_weak", 0),
    }


def evaluate_pandas(as_of: str | dt.date | None = None):
    """pandas DataFrame view (lazy import)."""
    import pandas as pd
    return pd.DataFrame(evaluate(as_of))


if __name__ == "__main__":
    c = compare_to_actuals()
    print("=== S&P methodology reimplementation vs actuals ===")
    print(f"as_of={c['as_of']} n_actuals={c['n_actuals_universe']} n_evaluated={c['n_evaluated']}")
    print(f"predicted S&P-eligible: {c['sp_predicted']}  actual S&P: {c['sp_actual']}")
    print(f"true positives: {c['true_positives']}")
    print(f"precision={c['precision']}  recall={c['recall']}  agreement={c['agreement']}")
    print(f"tier histogram: {c['tier_histogram']}")
    print(f"dual_strong={c['dual_strong']}  dual_weak={c['dual_weak']}")
    print("\nTop dual_strong by strength:")
    rows = [r for r in evaluate() if r["tier"] == "dual_strong"]
    rows.sort(key=lambda r: r["strength_score"], reverse=True)
    for r in rows[:12]:
        print(f"  {r['ticker']:6s} score={r['strength_score']:.3f} "
              f"de={r['debt_to_equity']} ic={r['interest_coverage']} "
              f"sp_actual={r['sp_actual_member']}")
