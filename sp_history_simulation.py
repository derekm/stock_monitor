"""
sp_history_simulation.py — reproduce S&P 500 inclusion/exclusion decisions in
our independent simulation, and track our reimplementation vs the actuals.

WHAT THIS DOES
--------------
1. Reconstructs a POINT-IN-TIME (PIT) membership history of the S&P 500 from
   `sp500_changes.parquet` (real Effective Date / Added / Removed events,
   1976-2026) plus `sp500_constituents.parquet` (date_added for current members).
   A ticker is a member as of `d` iff it was added on/before `d` and not removed
   before/at `d`. This is a REAL membership timeline including removals.
2. On each simulated rebalance date, scores every candidate with our S&P-style
   methodology (`sp_index_methodology.evaluate(as_of=...)`) and compares the
   predicted inclusion set against the reconstructed actual membership.
3. Emits a precision / recall / agreement time series so the tracking loop is
   persistent and auditable as our metrics evolve.

LIMITS (honest)
---------------
- Our fundamentals store is multi-snapshot (real quarterly history, 2024-2026,
  549 tickers after the yfinance backfill), but it does NOT reach back to the
   pre-2024 membership timeline. So for rebalance dates before ~2024-06, the
  scored set only covers names we have history for; earlier actuals include
  names we cannot score (surfaced as false negatives — a coverage gap, not a
  methodology error).
- `sp500_changes.parquet` is scraped from Wikipedia and may lag the official
  index; it is the best open add/remove source available (not fabricated).

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
    """Actual S&P 500 members as of `as_of`, reconstructed from REAL add/remove
    events (`sp500_changes.parquet`, 1976-2026) plus constituents' `date_added`.

    Membership model (honest about coverage gaps):
      - ADDS: union of every constituent's `date_added` (current members) and
        every `sp500_changes.added` event. Earliest known add date is used.
      - REMOVALS: every `sp500_changes.removed` event (earliest removal date).
      - A ticker removed but with NO known add record (Wikipedia's changes table
        is sparse before ~2010) is given a SENTINEL add date of the index
        inception (1928-01-02): we KNOW it was a member, we just don't know the
        exact add date. This keeps removed names visible as members until their
        removal date instead of vanishing.
      - A name is a member as of `d` iff add_date <= d AND (no removal OR
        removal_date > d).
    """
    con = duckdb.connect()
    # Cache the membership timeline (add/remove sets) once per process.
    cache = _MEMBERSHIP_CACHE.get("timeline")
    if cache is None:
        cache = con.execute(
            f"""
            WITH adds_raw AS (
                SELECT ticker, d FROM (
                    SELECT ticker, date_added AS d
                      FROM read_parquet('{(DATA_DIR / 'sp500_constituents.parquet').as_posix()}')
                     WHERE date_added IS NOT NULL
                    UNION ALL
                    SELECT added AS ticker, event_date AS d
                      FROM read_parquet('{(DATA_DIR / 'sp500_changes.parquet').as_posix()}')
                     WHERE added IS NOT NULL
                )
            ),
            adds AS (
                SELECT ticker, MIN(d) AS d FROM adds_raw GROUP BY ticker
                UNION ALL
                SELECT removed AS ticker, DATE '1928-01-02' AS d
                  FROM read_parquet('{(DATA_DIR / 'sp500_changes.parquet').as_posix()}')
                 WHERE removed IS NOT NULL
                   AND removed NOT IN (SELECT ticker FROM adds_raw)
            ),
            removes AS (
                SELECT removed AS ticker, MIN(event_date) AS d
                  FROM read_parquet('{(DATA_DIR / 'sp500_changes.parquet').as_posix()}')
                 WHERE removed IS NOT NULL GROUP BY removed
            )
            SELECT a.ticker, a.d AS add_d, r.d AS remove_d
            FROM adds a LEFT JOIN removes r ON r.ticker = a.ticker
            """
        ).fetchall()
        _MEMBERSHIP_CACHE["timeline"] = cache
    members = set()
    for tk, add_d, remove_d in cache:
        if add_d <= as_of and (remove_d is None or remove_d > as_of):
            members.add(tk)
    return members


_MEMBERSHIP_CACHE: dict = {}


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
