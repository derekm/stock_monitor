#!/usr/bin/env python3
"""granite_config.py — canonical Granite TTM configuration constants.

Leaf module (imports nothing from the repo) so every consumer can import
without circular-import risk. Consumers: granite_daily, granite_backfill,
window_padding, forecast_granite, pass4/pass5/pass6.

Values match the working venv's granite-tsfm 0.3.8 / transformers 5.14.1.
"""
DEFAULT_MODEL = "ibm-granite/granite-timeseries-ttm-r2"
CONTEXT = 512   # model context window (trading days)
HORIZON = 96    # model forecast horizon (trading days, TTM native ceiling)
BATCH = 8       # inference batch (granite_daily default)
