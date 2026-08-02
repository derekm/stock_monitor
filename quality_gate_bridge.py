"""
quality_gate_bridge.py — canonical dual-screen quality/value gate for stock_monitor.

Bridges the stockmagic analytics library (the "source of truth" for the
Buffett + trifecta + leverage gates) into the stock_monitor codebase so the
dual-screen analysis stops carrying its own copy-pasted, divergent thresholds.

Why this exists
---------------
`dual_screen_analysis.py` historically hardcoded Buffett as roe>=0.15 AND
roic>=0.15 AND de<=1.0 and a separate trifecta — which drifted from the
canonical gate in `stockmagic/src/analytics/quality_value.py` (Buffett
roe>=0.15 AND roic>=0.10; leverage de<=2.0 AND ic>=1.5; trifecta EV/EBITDA<=9,
P/B<=1.5, MktCap/Assets<=0.5). This bridge makes ONE gate the authority and
adds point-in-time (PIT) correctness: a screen "as of" a date only sees
fundamentals reported on or before that date.

Design notes
------------
- duckdb-only at import time (no pandas/numpy dependency) so it runs in the
  lean stockmagic venv and stays data-frame-agnostic. Callers that want a
  pandas/polars DataFrame can wrap `dual_screen_gate()` (returns list[dict])
  or call `dual_screen_gate_relation()` for the raw duckdb relation.
- Reads `fundamentals.parquet` (stock_monitor's PIT store), registers it under
  the column names the stockmagic gate expects, and reuses
  `quality_value.qualify` / `pit_snapshots.qualify_as_of` verbatim.

Requires stockmagic on PYTHONPATH (run from the stockmagic/ root, or the parent
env). The canonical gate lives at `src/analytics/quality_value.py` and
`src/data/pit_snapshots.py`.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent

# Make the stockmagic package importable regardless of CWD.
_ROOT = DATA_DIR.parent  # stockmagic/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics import quality_value as qv  # canonical gate + TRIFECTA
from src.data import pit_snapshots as pits  # PIT-aware qualify_as_of
from src.data.market_data import MarketDataStore


def _load_pit_store(as_of: dt.date) -> MarketDataStore:
    """Register stock_monitor's fundamentals.parquet as a PIT store and run the
    canonical, point-in-time qualify step. Returns the populated store."""
    store = MarketDataStore(":memory:")
    con = store.conn()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE pit_fundamentals AS
        SELECT ticker,
               as_of_date AS as_of,
               roe, roic,
               ev_ebitda,
               pb_ratio            AS pb,
               mktcap_to_assets    AS mcap_assets,
               debt_to_equity      AS debt_equity,
               interest_coverage
        FROM read_parquet('{(DATA_DIR / 'fundamentals.parquet').as_posix()}')
        """
    )
    # build_snapshot_timeseries keys PIT onto trading dates from daily_prices,
    # so register a minimal daily_prices (one row per distinct as_of_date) to
    # give the PIT series its date axis. The gate itself is fundamentals-only.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE daily_prices AS
        SELECT DISTINCT ticker, as_of_date AS trade_date, NULL AS adj_close
        FROM read_parquet('{(DATA_DIR / 'fundamentals.parquet').as_posix()}')
        """
    )
    # Canonical PIT gate: only fundamentals with as_of <= date are visible.
    # qualify_as_of reads from pit_snapshots, so build the dated series first.
    pits.build_snapshot_timeseries(store, recompute_marketcap=False)
    pits.qualify_as_of(store, as_of)
    return store


def _monitored_set() -> set[str]:
    import duckdb
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT ticker FROM read_parquet('{(DATA_DIR / 'monitored_stocks.parquet').as_posix()}')"
    ).fetchall()
    return {r[0] for r in rows}


def dual_screen_gate(as_of: str | dt.date | None = None) -> list[dict]:
    """Return the dual-screen gap as a list of row dicts using the canonical
    stockmagic gate.

    Parameters
    ----------
    as_of : point-in-time date. None => use the latest available fundamentals
            for each ticker (equivalent to the old tail(1) behaviour).

    Each row: ticker, monitored, buffett_pass, trifecta_pass, gap,
    roe, roic, debt_to_equity, pb_ratio, ev_ebitda, mktcap_to_assets.
    """
    if as_of is None:
        import duckdb
        con = duckdb.connect()
        as_of = con.execute(
            f"SELECT MAX(as_of_date) FROM read_parquet('{(DATA_DIR / 'fundamentals.parquet').as_posix()}')"
        ).fetchone()[0]
    if isinstance(as_of, str):
        as_of = dt.date.fromisoformat(as_of)

    store = _load_pit_store(as_of)
    qp = store.conn().execute("SELECT * FROM quality_pass").fetchall()
    cols = [d[0] for d in store.conn().description]
    monitored = _monitored_set()

    rows = []
    for rec in qp:
        r = dict(zip(cols, rec))
        tk = r["ticker"]
        buffett = bool(r["buffett_ok"]) and bool(r["leverage_ok"])
        trif = bool(r["trifecta_ok"])
        gap = (
            "dual" if (buffett and trif) else
            "quality_only" if buffett else
            "value_only" if trif else
            "neither"
        )
        rows.append({
            "ticker": tk,
            "monitored": tk in monitored,
            "buffett_pass": buffett,
            "trifecta_pass": trif,
            "gap": gap,
            "roe": r.get("roe"), "roic": r.get("roic"),
            "debt_to_equity": r.get("debt_equity"),
            "pb_ratio": r.get("pb"), "ev_ebitda": r.get("ev_ebitda"),
            "mktcap_to_assets": r.get("mcap_assets"),
        })
    return rows


def dual_screen_gate_relation(as_of: str | dt.date | None = None):
    """Return the raw duckdb relation (for SQL/Chart.js/polars callers)."""
    if as_of is None:
        import duckdb
        con = duckdb.connect()
        as_of = con.execute(
            f"SELECT MAX(as_of_date) FROM read_parquet('{(DATA_DIR / 'fundamentals.parquet').as_posix()}')"
        ).fetchone()[0]
    if isinstance(as_of, str):
        as_of = dt.date.fromisoformat(as_of)
    store = _load_pit_store(as_of)
    return store.conn().execute("SELECT * FROM quality_pass")


def dual_screen_summary(as_of: str | dt.date | None = None) -> dict:
    """Convenience summary counts (mirrors dual_screen_analysis.print output)."""
    gap = dual_screen_gate(as_of)
    return {
        "as_of": str(as_of) if as_of else "latest",
        "n": len(gap),
        "buffett_pass": sum(r["buffett_pass"] for r in gap),
        "trifecta_pass": sum(r["trifecta_pass"] for r in gap),
        "dual": sum(r["buffett_pass"] and r["trifecta_pass"] for r in gap),
    }


def dual_screen_gate_pandas(as_of: str | dt.date | None = None):
    """pandas DataFrame view (lazy import — only if pandas is available)."""
    import pandas as pd
    return pd.DataFrame(dual_screen_gate(as_of))


if __name__ == "__main__":
    s = dual_screen_summary()
    print("=== Canonical dual-screen gate (PIT-aware, shared with stockmagic) ===")
    print(f"as_of={s['as_of']}  n={s['n']}")
    print(f"Buffett quality pass: {s['buffett_pass']}")
    print(f"Value trifecta pass:  {s['trifecta_pass']}")
    print(f"BOTH:                 {s['dual']}")
    gap = dual_screen_gate()
    print(f"\nGap rows: {len(gap)} (showing first 15)")
    for r in gap[:15]:
        print(f"  {r['ticker']:6s} {r['gap']:12s} "
              f"roe={r['roe']} roic={r['roic']} de={r['debt_to_equity']} "
              f"pb={r['pb_ratio']} ev={r['ev_ebitda']} mca={r['mktcap_to_assets']}")
