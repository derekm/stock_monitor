"""
sp_history_simulation.py — reproduce S&P 500 inclusion/exclusion decisions in
our independent simulation, and track our reimplementation vs the actuals.

WHAT THIS DOES (foundation)
---------------------------
1. Reconstructs a POINT-IN-TIME (PIT) membership history of the S&P 500 from
   `sp500_constituents.parquet` (`date_added`). A current member is treated as
   IN the index from its `date_added` onward. This is a real, if addition-only,
   membership timeline (we do not yet have removal/ deletion events; see LIMITS).
2. On each simulated rebalance date, scores every candidate with our S&P-style
   methodology (`sp_index_methodology.evaluate(as_of=...)`) and compares the
   predicted inclusion set against the reconstructed actual membership.
3. Emits a precision / recall / agreement time series so the tracking loop is
   persistent and auditable as our metrics evolve.

LIMITS (honest)
---------------
- Our fundamentals store is a SINGLE snapshot (PIT-backfilled, constant over
  time), so per-date fundamentals don't yet vary. The simulation therefore
  shows our *current* methodology applied at each past date, not the methodology
  we would have run *with then-available data*. Multi-snapshot fundamentals
  (see pit_snapshots TODO) are the prerequisite for a fully historical replay.
- Removal events (companies dropped from the index) are not yet modeled; actual
  membership is addition-only. Precision is exact; recall is a lower bound.
- We only hold fundamentals for 142 names, so most historical constituents are
  predicted-False (false negatives) — a coverage gap, surfaced per date.

Design: duckdb-only at import (pandas optional). Reads stock_monitor data files.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).parent

# Reuses the PIT-aware qualify machinery from stockmagic (via the bridge module)
from sp_index_methodology import pits  # noqa: E402  (pits = pit_snapshots)

# Rebalance cadence to simulate (S&P rebalances continuously, but we sample
# quarterly quarter-ends as decision dates for tractability).
REBALANCE_QUARTER_ENDS = True


def pit_membership_as_of(as_of: dt.date) -> set[str]:
    """Actual S&P 500 members as of `as_of`, reconstructed from date_added.

    A current constituent is a member iff date_added <= as_of."""
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT ticker FROM read_parquet('{(DATA_DIR / 'sp500_constituents.parquet').as_posix()}')
        WHERE current AND date_added IS NOT NULL AND date_added <= DATE '{as_of}'
        """
    ).fetchall()
    return {r[0] for r in rows}


def rebalance_dates(start: dt.date, end: dt.date):
    """Quarter-end decision dates in [start, end]."""
    d = dt.date(start.year, start.month, 1)
    out = []
    while d <= end:
        q = (d.month - 1) // 3
        qe_month = q * 3 + 3
        year = d.year + (qe_month > 12)
        month = qe_month % 12 or 12
        last = dt.date(year, month, 1) - dt.timedelta(days=1) if False else None
        # last day of quarter-end month
        if month == 12:
            last = dt.date(year, 12, 31)
        elif month == 3:
            last = dt.date(year, 3, 31)
        elif month == 6:
            last = dt.date(year, 6, 30)
        else:
            last = dt.date(year, 9, 30)
        if last >= start:
            out.append(last)
        # advance 3 months
        nm = month + 3
        ny = year
        if nm > 12:
            nm -= 12
            ny += 1
        d = dt.date(ny, nm, 1)
    return out


def simulate(start: str = "2015-01-01", end: str | None = None) -> list[dict]:
    """Run the historical simulation. Returns one row per rebalance date with
    the tracking metrics (precision/recall/agreement, counts, tier histogram).

    The PIT snapshot series is built ONCE (expensive over the 1962->2026 price
    history); each rebalance date only re-runs the cheap qualify_as_of + the
    S&P-style prediction.
    """
    import sp_index_methodology as sim
    from sp_index_methodology import _store, _strength_score, LEV_STRONG_DE_MAX, LEV_STRONG_IC_MIN

    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end) if end else dt.date.today()
    dates = rebalance_dates(s, e)

    store = _store()           # build pit_fundamentals/daily_prices/monitored + snapshots once
    con = store.conn()
    # monitored lookup (static)
    mon = {}
    for rec in con.execute(
        "SELECT ticker, sp500_member, sp500_sector, market_cap FROM monitored_stocks"
    ).fetchall():
        mon[rec[0]] = {"sp500_member": bool(rec[1]), "sp_sector": rec[2], "mcap": rec[3]}

    out = []
    for d in dates:
        pits.qualify_as_of(store, d)
        qp = con.execute("SELECT * FROM quality_pass").fetchall()
        cols = [c[0] for c in con.description]
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
            m = mon.get(tk, {})
            mcap = m.get("mcap")
            sp_sector = m.get("sp_sector")
            roe = r.get("roe")
            sp_eligible = (
                (mcap is not None and mcap >= 1.0e10)
                and (roe is not None and roe > 0)
                and (sp_sector is not None)
            )
            rows.append({"ticker": tk, "sp_predicted_member": sp_eligible,
                         "sp_actual_member": bool(m.get("sp500_member", False))})
        pred = {r["ticker"] for r in rows if r["sp_predicted_member"]}
        actual = pit_membership_as_of(d)
        tp = len(actual & pred)
        fp = len(pred - actual)
        fn = len(actual - pred)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        agreement = tp / len(actual) if actual else None
        out.append({
            "date": str(d),
            "n_actual": len(actual),
            "n_predicted": len(pred),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "agreement": agreement,
        })
    return out


def simulate_pandas(start: str = "2015-01-01", end: str | None = None):
    import pandas as pd
    return pd.DataFrame(simulate(start, end))


if __name__ == "__main__":
    res = simulate()
    print(f"=== Historical S&P inclusion/exclusion simulation ({len(res)} rebalance dates) ===")
    print("date        n_actual n_pred  TP  FP  FN   prec   recall  agree")
    for r in res:
        p = f"{r['precision']:.2f}" if r["precision"] is not None else "  - "
        rc = f"{r['recall']:.2f}" if r["recall"] is not None else "  - "
        ag = f"{r['agreement']:.2f}" if r["agreement"] is not None else "  - "
        print(f"{r['date']}  {r['n_actual']:7d} {r['n_predicted']:5d} "
              f"{r['true_positives']:4d} {r['false_positives']:3d} {r['false_negatives']:4d}  "
              f"{p:>5}  {rc:>5}  {ag:>5}")
    print("\nNOTE: fundamentals are a single PIT-backfilled snapshot, so per-date")
    print("scores are constant; recall is a coverage lower bound (we hold 142 names).")
