# ttm_exogenous.py

Exogenous channels for Granite TTM: market EW return, 20d vol, cross-sectional dispersion, sector EW returns; optional external CSV.

```bash
python ttm_exogenous.py --save
python ttm_exogenous.py --from-csv macro.csv --save
```

Output: `exogenous_panel.parquet`. Use with `forecast_granite.py --exog`.
