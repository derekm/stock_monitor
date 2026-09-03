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


def hf_model_cached(model_name: str = DEFAULT_MODEL) -> bool:
    """True when the HF snapshot for `model_name` is already on disk.

    from_pretrained revalidates against the hub by default; on this machine
    that request has been observed to hang for minutes with no progress. When
    the snapshot exists, pass local_files_only=True instead (daily cron hung
    on granite_daily.py model load; fix 2026-09-03)."""
    from pathlib import Path
    slug = "models--" + str(model_name).replace("/", "--")
    return (Path.home() / ".cache" / "huggingface" / "hub" / slug).is_dir()
