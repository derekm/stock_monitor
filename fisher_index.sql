-- Chained Fisher / Laspeyres / Paasche helpers for DuckDB
-- Requires a long table: date, ticker, price (close), qty (volume)
-- Precomputed results live in fisher_indexes (dashboard + fisher_indexes.csv)

-- Latest snapshot by universe
SELECT universe, freq,
       max(date) AS as_of,
       arg_max(fisher_p, date) AS fisher_p,
       arg_max(fisher_q, date) AS fisher_q,
       arg_max(nominal_sqrt_fisher, date) AS nominal_sqrt_fisher,
       arg_max(nominal_fisher_product, date) AS nominal_fisher_product
FROM fisher_indexes
GROUP BY 1, 2
ORDER BY 1, 2;

-- Portfolio daily path
SELECT date, laspeyres_p, paasche_p, fisher_p, fisher_q,
       nominal_sqrt_fisher, nominal_fisher_product, n_items
FROM fisher_indexes
WHERE universe = 'portfolio' AND freq = 'D'
ORDER BY date;

-- Materials sector
SELECT date, fisher_p, fisher_q, nominal_sqrt_fisher
FROM fisher_indexes
WHERE universe = 'Materials'
ORDER BY date;

-- Definitions (period link t-1 → t):
-- Laspeyres_P = sum(p_t * q_{t-1}) / sum(p_{t-1} * q_{t-1})
-- Paasche_P   = sum(p_t * q_t)     / sum(p_{t-1} * q_t)
-- Fisher_P    = sqrt(Laspeyres_P * Paasche_P)
-- Laspeyres_Q = sum(p_{t-1} * q_t) / sum(p_{t-1} * q_{t-1})
-- Paasche_Q   = sum(p_t * q_t)     / sum(p_t * q_{t-1})
-- Fisher_Q    = sqrt(Laspeyres_Q * Paasche_Q)
-- Nominal product link = Fisher_P * Fisher_Q
-- Nominal geometric    = sqrt(Fisher_P * Fisher_Q)
-- Chained level = 100 * cumprod(links)
