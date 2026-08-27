#!/usr/bin/env python3
"""
window_padding.py — fill a sub-512 context for short-history tickers.

The Granite TTM-r2 has a FIXED 512-token context; it cannot be shrunk. New S&P
additions (spinoffs like HONA/SPCX/SPR, recent IPOs) often have < 512 trading
days. To give them a real (non-fabricated) 512-window we PAD THE HEAD with a
genuine market proxy, rescaled to the ticker's own price level:

  padded_context = [ proxy_rescaled_to_ticker_level (first 512-n),
                     ticker_actual_close (last n) ]

Proxy choice (best-first):
  1. Same-GICS-sector equal-weight average close (real regime shape: 2008/2020/22).
  2. Cross-sectional market mean close (for tickers with no GICS sector tag).

The rescale matches the proxy's level to the ticker's first available close so
the transition into the real ticker data is continuous. This is the only
non-invented way to feed a fixed-context model; the forecast head still trains
on the ticker's OWN realized moves, so accuracy compounds as real days accrue.

When a ticker later accumulates >= 512 real days, callers should switch it to a
pure (unpadded) 512 window and run a fresh backfill -- see needs_backfill().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from granite_config import CONTEXT, HORIZON  # canonical Granite config (leaf)

PRICES = "daily_prices/"
CONSTITS = "sp500_constituents.parquet"


def _load():
    px = pd.read_parquet(PRICES)
    try:
        cons = pd.read_parquet(CONSTITS)
    except Exception:
        cons = None
    return px, cons


def _sector_proxy(px: pd.DataFrame, sector: str | None, dates, proxy_cache: dict | None = None) -> np.ndarray | None:
    """Equal-weight daily average close for a GICS sector (or whole market),
    aligned to `dates`. Returns raw proxy series or None."""
    cache_key = sector or "__MARKET__"
    if proxy_cache is not None and cache_key in proxy_cache:
        cached = proxy_cache[cache_key]
        return np.asarray(cached, dtype=np.float32)
    try:
        cons = pd.read_parquet(CONSTITS)
    except Exception:
        return None
    if cons is None or cons.empty:
        return None
    if sector:
        tickers = cons.loc[cons["gics_sector"] == sector, "ticker"].tolist()
    else:
        tickers = cons["ticker"].tolist()
    sub = px[px["ticker"].isin(tickers)]
    if sub.empty:
        return None
    # equal-weight average per date
    avg = sub.groupby("date")["close"].mean()
    avg = avg.reindex(pd.to_datetime(dates)).ffill().bfill()
    if avg.isna().all():
        return None
    out = avg.values.astype(np.float32)
    if proxy_cache is not None:
        proxy_cache[cache_key] = out
    return out


def pad_to_context(ticker: str, close: np.ndarray, sector: str | None = None,
                   px: pd.DataFrame | None = None, cons: pd.DataFrame | None = None,
                   proxy_cache: dict | None = None,
                   last_date=None, all_dates=None) -> np.ndarray:
    """Return a (CONTEXT,) array: proxy-padded head + real ticker tail.
    `close` is the ticker's actual closes (any length >= 1). The proxy is built
    over the last CONTEXT trading dates (ending at the ticker's last date) so the
    padded head has genuine market history, then rescaled to the ticker's first
    close level for a continuous transition into the real tail.

    `proxy_cache` (optional) maps sector-name -> precomputed proxy ndarray and
    short-circuits the repeated parquet re-reads that used to happen per-window
    (the old code called pd.read_parquet(CONSTITS) inside _sector_proxy on every
    call). Pass a shared dict to eliminate thousands of redundant file reads.

    `last_date` / `all_dates` (optional, precomputed by the caller) skip the
    O(N) per-window full-frame scans (px[px["ticker"]==ticker] and
    set(px["ticker"].unique())) that used to make short-ticker padding the
    dominant CPU cost when many windows share one ticker. When omitted, the old
    behavior is preserved (scans px each call)."""
    if proxy_cache is None:
        proxy_cache = {}
    n = len(close)
    if n >= CONTEXT:
        return close[-CONTEXT:].astype(np.float32)
    if px is None or cons is None:
        px, cons = _load()
    # last CONTEXT trading dates across the universe, ending at ticker's last date
    if last_date is None or all_dates is None:
        last_date = pd.to_datetime(px[px["ticker"] == ticker]["date"]).max() if ticker in set(px["ticker"].unique()) else pd.to_datetime(px["date"]).max()
        all_dates = pd.to_datetime(px["date"]).sort_values().drop_duplicates()
    window_dates = all_dates[all_dates <= last_date]
    if len(window_dates) > CONTEXT:
        window_dates = window_dates[-CONTEXT:]  # last CONTEXT dates (engine-agnostic; works for numpy + pandas)
    proxy = _sector_proxy(px, sector, window_dates, proxy_cache=proxy_cache)
    if proxy is None or np.all(np.isnan(proxy)) or len(proxy) < CONTEXT:
        # fall back to market proxy, or flat-pad if truly unavailable
        proxy = _sector_proxy(px, None, window_dates, proxy_cache=proxy_cache)
    if proxy is None or np.all(np.isnan(proxy)) or len(proxy) < CONTEXT:
        head = np.full(CONTEXT - n, float(close[0]), dtype=np.float32)
        return np.concatenate([head, close.astype(np.float32)])
    # proxy is length CONTEXT (one value per window date)
    head_proxy = proxy[: CONTEXT - n]
    # rescale so the proxy's LAST head value matches the ticker's first real
    # close -> continuous handoff into the real tail (no fabricated level jump)
    j = CONTEXT - n - 1
    scale = float(close[0]) / float(head_proxy[j]) if head_proxy[j] != 0 else 1.0
    head = (head_proxy * scale).astype(np.float32)
    return np.concatenate([head, close.astype(np.float32)])


def needs_backfill(ticker: str, px: pd.DataFrame | None = None) -> tuple[bool, int]:
    """A ticker that was padded should be re-backfilled (unpadded) once it has
    >= CONTEXT real days. Returns (should_backfill_now, current_real_days)."""
    if px is None:
        px = pd.read_parquet(PRICES)
    n = int((px["ticker"] == ticker).sum())
    return (n >= CONTEXT), n
