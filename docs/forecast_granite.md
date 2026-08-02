# forecast_granite.py

Stock (and sector) forecasting with **IBM Granite TTM**, with statistical fallback when weights are offline.

## Usage
```bash
python forecast_granite.py status
python forecast_granite.py forecast --index portfolio --from-first-trade --horizon 10
python forecast_granite.py forecast --index portfolio --multivariate --exog --horizon 10
python forecast_granite.py forecast --index sectors --horizon 10
python forecast_granite.py forecast --sector Materials,Energy --horizon 10
python forecast_granite.py forecast --ticker MOS --channels full --rolling --horizon 30
python forecast_granite.py backtest --index portfolio --horizon 10 --window 40
```

## Features
- First-trade history anchors for portfolio names
- Multivariate peer channels; `--exog` market/sector exogenous panel
- Rolling iterative long-horizon forecasts
- Metrics: MAE, RMSE, MAPE, directional accuracy

## Install for real TTM weights
```bash
pip install granite-tsfm transformers torch accelerate
```

See also: [ttm_features.md](ttm_features.md), [ttm_exogenous.md](ttm_exogenous.md), [analyze_granite_forecasts.md](analyze_granite_forecasts.md).


## Multi-index runs

```bash
python forecast_granite.py forecast --index portfolio,defensive,growth --horizon 10
python forecast_granite.py backtest --index portfolio --index growth --horizon 10 --window 40
python analyze_granite_forecasts.py --index portfolio,defensive
```

Outputs include `index_name` (comma-joined when a ticker belongs to multiple requested indexes).
