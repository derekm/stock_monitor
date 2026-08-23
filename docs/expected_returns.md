# expected_returns.py

Ilmanen 4-pillar expected-return ranks on the `daily_prices` universe.

## Pillars

| Pillar | Definition |
|--------|------------|
| carry | Cross-sectional rank of earnings yield (`net_income_ttm` / `market_cap`) and FCF yield |
| value | Mean rank of B/M, E/P, FCF yield, S/P using fundamentals / daily `market_cap` |
| momentum | Rank of 12-1 return (252d, skip last 21d) |
| defensive | Mean rank of −60d vol, −|beta| vs EW market, ROE, ROIC, −D/E |
| expected_return | Equal-weight mean of available pillars |

## Usage

```bash
python expected_returns.py --save
```

Reads snapshots of `daily_prices.parquet` and `fundamentals.parquet` (does not hold the live files). Writes month-end long `expected_returns_decomp.parquet`: `date`, `ticker`, pillar ranks, `expected_return`.
