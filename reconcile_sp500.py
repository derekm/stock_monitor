"""Reconcile sp500_member / sp500_sector / sp500_date_added in
monitored_stocks.parquet against the authoritative sp500_constituents.parquet.

The authoritative list is the Wikipedia-derived current S&P 500 (U.S.-listed
common stocks only). stock_monitor (built by a prior assistant) incorrectly
carried ADRs/ETFs and stale flags into sp500_member; we overwrite those three
columns from the canonical list. All other columns are preserved untouched.
"""
from __future__ import annotations
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
MON = HERE / "monitored_stocks.parquet"
CONST = HERE / "sp500_constituents.parquet"
OUT = HERE / "monitored_stocks.parquet"  # overwrite in place


def main():
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE monitored AS
        SELECT * FROM read_parquet('{MON.as_posix()}')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE sp500 AS
        SELECT ticker, gics_sector AS sp500_sector_new,
               date_added AS sp500_date_added
        FROM read_parquet('{CONST.as_posix()}')
        """
    )
    # Overwrite the three S&P columns from the authoritative table.
    con.execute(
        """
        CREATE OR REPLACE TABLE monitored_new AS
        SELECT
            m.* EXCLUDE (sp500_member, sp500_sector),
            (s.ticker IS NOT NULL)                    AS sp500_member,
            s.sp500_sector_new                        AS sp500_sector,
            s.sp500_date_added                       AS sp500_date_added
        FROM monitored m
        LEFT JOIN sp500 s ON s.ticker = m.ticker
        """
    )
    con.execute(
        f"COPY (SELECT * FROM monitored_new) TO '{OUT.as_posix()}' (FORMAT PARQUET, OVERWRITE true)"
    )
    print("wrote", OUT)
    print("sp500_member dist:",
          con.execute("SELECT sp500_member, COUNT(*) FROM monitored_new GROUP BY 1").fetchall())
    print("sp500_sector non-null:",
          con.execute("SELECT COUNT(*) FROM monitored_new WHERE sp500_sector IS NOT NULL").fetchone())
    print("overlap monitored∩S&P:",
          con.execute("SELECT COUNT(*) FROM monitored_new WHERE sp500_member").fetchone())


if __name__ == "__main__":
    main()
