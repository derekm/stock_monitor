#!/usr/bin/env python3
"""
run_fisher_duckdb.py — Chained Fisher / Laspeyres / Paasche indexes computed in DuckDB.

p = close, q = volume (ffilled). Chained levels base = 100.

Usage:
  python run_fisher_duckdb.py --universe portfolio --save
  python run_fisher_duckdb.py --sector Materials --save
  python run_fisher_duckdb.py --tickers MOS,CF,NTR --save
  python run_fisher_duckdb.py --universe fertilizer --freq W --save
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from cli_common import (
    add_index_args, add_ticker_args, add_sector_arg, add_save_arg, add_freq_arg,
    add_window_arg, resolve_tickers_from_args, resolve_index_names_from_args,
    build_parser,
)
from index_registry import parse_indexes, tickers_for_index, available_indexes, index_help_text

DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices.parquet"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
OUT_CSV = DATA_DIR / "fisher_indexes_duckdb.parquet"
OUT_PQ = DATA_DIR / "fisher_indexes_duckdb.parquet"

CORE_SQL = r"""
CREATE OR REPLACE TEMP TABLE pq AS
SELECT
  date,
  ticker,
  price,
  COALESCE(
    last_value(qty IGNORE NULLS) OVER (
      PARTITION BY ticker ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ),
    1.0
  ) AS qty
FROM price_qty
WHERE ticker IN (SELECT ticker FROM tickers)
  AND price IS NOT NULL AND price > 0;

CREATE OR REPLACE TEMP TABLE date_pairs AS
WITH c AS (SELECT DISTINCT date FROM pq)
SELECT date AS date_t, lag(date) OVER (ORDER BY date) AS date_0
FROM c;

CREATE OR REPLACE TEMP TABLE period_links AS
SELECT
  d.date_t AS date,
  d.date_0,
  sum(p1.price * p0.qty) AS sum_p1q0,
  sum(p0.price * p0.qty) AS sum_p0q0,
  sum(p1.price * p1.qty) AS sum_p1q1,
  sum(p0.price * p1.qty) AS sum_p0q1,
  count(*)::BIGINT AS n_items
FROM date_pairs d
JOIN pq p0 ON p0.date = d.date_0
JOIN pq p1 ON p1.date = d.date_t AND p1.ticker = p0.ticker
WHERE d.date_0 IS NOT NULL AND p0.qty > 0 AND p1.qty > 0
GROUP BY 1, 2;

CREATE OR REPLACE TEMP TABLE links AS
SELECT
  date,
  n_items,
  CASE WHEN sum_p0q0 > 0 THEN sum_p1q0 / sum_p0q0 ELSE 1.0 END AS link_lp,
  CASE WHEN sum_p0q1 > 0 THEN sum_p1q1 / sum_p0q1 ELSE 1.0 END AS link_pp,
  CASE WHEN sum_p0q0 > 0 THEN sum_p0q1 / sum_p0q0 ELSE 1.0 END AS link_lq,
  CASE WHEN sum_p1q0 > 0 THEN sum_p1q1 / sum_p1q0 ELSE 1.0 END AS link_pq
FROM period_links;

CREATE OR REPLACE TEMP TABLE links_f AS
SELECT
  date,
  n_items,
  link_lp, link_pp,
  CASE WHEN link_lp > 0 AND link_pp > 0 THEN sqrt(link_lp * link_pp) ELSE 1.0 END AS link_fp,
  link_lq, link_pq,
  CASE WHEN link_lq > 0 AND link_pq > 0 THEN sqrt(link_lq * link_pq) ELSE 1.0 END AS link_fq
FROM links;

CREATE OR REPLACE TEMP TABLE fisher_chained AS
WITH base AS (
  SELECT min(date_0) AS date FROM period_links
),
seed AS (
  SELECT date, CAST(NULL AS BIGINT) AS n_items,
         1.0 AS link_lp, 1.0 AS link_pp, 1.0 AS link_fp,
         1.0 AS link_lq, 1.0 AS link_pq, 1.0 AS link_fq
  FROM base
  UNION ALL
  SELECT date, n_items, link_lp, link_pp, link_fp, link_lq, link_pq, link_fq
  FROM links_f
),
ch AS (
  SELECT
    date,
    n_items,
    link_lp, link_pp, link_fp, link_lq, link_pq, link_fq,
    100 * exp(sum(ln(greatest(link_lp, 1e-12))) OVER w) AS laspeyres_p,
    100 * exp(sum(ln(greatest(link_pp, 1e-12))) OVER w) AS paasche_p,
    100 * exp(sum(ln(greatest(link_fp, 1e-12))) OVER w) AS fisher_p,
    100 * exp(sum(ln(greatest(link_lq, 1e-12))) OVER w) AS laspeyres_q,
    100 * exp(sum(ln(greatest(link_pq, 1e-12))) OVER w) AS paasche_q,
    100 * exp(sum(ln(greatest(link_fq, 1e-12))) OVER w) AS fisher_q
  FROM seed
  WINDOW w AS (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
SELECT
  date,
  laspeyres_p, paasche_p, fisher_p,
  laspeyres_q, paasche_q, fisher_q,
  fisher_p * fisher_q AS nominal_fisher_product,
  sqrt(greatest(fisher_p, 0) * greatest(fisher_q, 0)) AS nominal_sqrt_fisher,
  link_lp AS link_laspeyres_p,
  link_pp AS link_paasche_p,
  link_fp AS link_fisher_p,
  link_lq AS link_laspeyres_q,
  link_pq AS link_paasche_q,
  link_fq AS link_fisher_q,
  n_items
FROM ch
ORDER BY date;
"""


def resolve_tickers(universe=None, sector=None, tickers=None) -> list[str]:
    if tickers:
        return [x.strip().upper() for x in tickers.split(",") if x.strip()]
    if sector:
        stocks = pd.read_parquet(STOCKS) if Path(STOCKS).exists() else pd.DataFrame()
        if not stocks.empty and "sector" in stocks.columns:
            return stocks.loc[stocks["sector"].str.lower() == sector.lower(), "ticker"].tolist()
    if universe:
        try:
            names = parse_indexes(universe)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        seen, out = set(), []
        for n in names:
            for tk in tickers_for_index(n):
                if tk not in seen:
                    seen.add(tk)
                    out.append(tk)
        return out
    return tickers_for_index("all")



def compute(tickers: list[str], freq: str = "D", label: str = "") -> pd.DataFrame:
    con = duckdb.connect()
    tickers_sql = ", ".join(repr(t) for t in tickers)
    con.execute(f"""
      CREATE OR REPLACE TABLE raw AS
      SELECT CAST(date AS DATE) AS date,
             ticker,
             CAST(close AS DOUBLE) AS price,
             CASE WHEN volume IS NULL OR volume <= 0 THEN NULL
                  ELSE CAST(volume AS DOUBLE) END AS qty
      FROM read_parquet('{PRICES.as_posix()}')
      WHERE ticker IN ({tickers_sql});
    """)
    if freq.upper() == "W":
        con.execute("""
          CREATE OR REPLACE TABLE price_qty AS
          SELECT date_trunc('week', date)::DATE AS date, ticker,
                 arg_max(price, date) AS price, sum(qty) AS qty
          FROM raw GROUP BY 1, 2;
        """)
    elif freq.upper() == "M":
        con.execute("""
          CREATE OR REPLACE TABLE price_qty AS
          SELECT date_trunc('month', date)::DATE AS date, ticker,
                 arg_max(price, date) AS price, sum(qty) AS qty
          FROM raw GROUP BY 1, 2;
        """)
    else:
        con.execute("CREATE OR REPLACE TABLE price_qty AS SELECT * FROM raw;")

    con.execute("CREATE OR REPLACE TABLE tickers AS SELECT * FROM (SELECT unnest(?::VARCHAR[]) AS ticker)", [tickers])
    con.execute(CORE_SQL)
    df = con.execute("SELECT * FROM fisher_chained ORDER BY date").df()
    df["universe"] = label or "custom"
    df["freq"] = freq.upper()
    df["n_tickers"] = len(tickers)
    con.close()
    return df


def main():
    ap = argparse.ArgumentParser(description="Chained Fisher indexes in DuckDB")
    add_index_args(ap, default="portfolio")
    add_ticker_args(ap)
    add_sector_arg(ap)
    add_freq_arg(ap)
    add_save_arg(ap)
    args = ap.parse_args()

    tickers = resolve_tickers_from_args(args, default_index='portfolio')
    if not tickers:
        raise SystemExit("No tickers resolved")
    label = args.sector or (",".join(resolve_index_names_from_args(args, default_index="portfolio")) or "custom")
    print(f"DuckDB Fisher · {len(tickers)} names · {label} · freq={args.freq}")
    df = compute(tickers, freq=args.freq, label=label)
    cols = [c for c in ["date", "fisher_p", "fisher_q", "nominal_sqrt_fisher", "n_items"] if c in df.columns]
    print(df[cols].tail(8).to_string(index=False))
    print(f"Last: Fisher_P={df['fisher_p'].iloc[-1]:.2f}  Fisher_Q={df['fisher_q'].iloc[-1]:.2f}  "
          f"√(Fp·Fq)={df['nominal_sqrt_fisher'].iloc[-1]:.2f}")

    if args.save:
        if OUT_CSV.exists():
            old = pd.read_parquet(OUT_CSV)
            # date column is a DATE; read as string then ingest as datetime.date
            old["date"] = old["date"].apply(
                lambda s: datetime.strptime(str(s)[:10], "%Y-%m-%d").date())
            old = old[~((old["universe"] == label) & (old["freq"] == args.freq.upper()))]
            out = pd.concat([old, df], ignore_index=True)
        else:
            out = df
        out.to_parquet(OUT_CSV, index=False)
        try:
            out.to_parquet(OUT_PQ, index=False)
        except Exception:
            pass
        print(f"Wrote {OUT_CSV} ({len(out)} rows)")


if __name__ == "__main__":
    main()
