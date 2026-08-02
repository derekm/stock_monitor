# ttm_features.py

Build TTM-ready multivariate panels from `daily_prices.parquet`.

## Channels
close, volume, log returns, 20d vol, RSI-14, MA ratio, HL range, volume z-score

## Usage
```bash
python ttm_features.py --index portfolio --mode close_only --save
python ttm_features.py --ticker MOS --save
```

Panels land under `ttm_panels/` for Granite multivariate forecasting.
