# expected_returns.py

Ilmanen 4-pillar expected-return ranks on the `daily_prices` universe.

## Pillars

| Pillar | Definition |
|--------|------------|
| carry | Rank of EY + FCF yield. Mcap = daily `market_cap`, else `shares_outstanding × adj_close`, else fund `market_cap`. |
| value | Mean rank of B/M, E/P, FCF yield, S/P on the same mcap |
| momentum | Rank of 12-1 return (252d, skip last 21d) |
| defensive | Mean rank of −60d vol, −|beta| vs EW market, ROE, ROIC, −D/E |
| expected_return | Equal-weight mean of available pillars; **NaN unless ≥2 pillars, `instrument_type=stock`, and carry or value is present** |

## Usage

```bash
python expected_returns.py --save
```

Reads snapshots of `daily_prices.parquet` and `fundamentals.parquet` (does not hold the live files). Writes month-end long `expected_returns_decomp.parquet`: `date`, `ticker`, pillar ranks, `expected_return`.

## CF vs DR (`implied_r_decomp.parquet`)

`cf_yield = ROE / P/B` (clean-surplus earnings yield). `discount_rate = 2·ROE/(P/B+1)` (RIV). Gap = CF − DR.

## Regime premia

`python factor_library.py --regime-premia --save` → `regime_factor_premia.parquet` (HMM × available FF columns).
