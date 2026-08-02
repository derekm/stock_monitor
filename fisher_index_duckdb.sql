-- Chained Fisher indexes in DuckDB (p=price/close, q=volume)
-- Prerequisites:
--   CREATE TABLE price_qty(date DATE, ticker VARCHAR, price DOUBLE, qty DOUBLE);
--   CREATE TABLE tickers(ticker VARCHAR);
-- Then run the statements below (or: python run_fisher_duckdb.py ...)


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


-- Inspect
SELECT date, fisher_p, fisher_q, nominal_sqrt_fisher, nominal_fisher_product, n_items
FROM fisher_chained ORDER BY date DESC LIMIT 10;
