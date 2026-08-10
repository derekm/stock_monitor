#!/usr/bin/env python3
"""analytics_common.py — shared Polars/pandas loaders and return helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import scipy.optimize  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DATA_DIR = Path(__file__).resolve().parent

# ── Canonical dual-pass / INCLUDE_CORE thresholds ───────────────────────────
# Single source of truth for the quality/value dual-pass gates. Consumers:
# preferred_metrics.py, fundamentals_history.py, threshold_logic.py,
# inclusion_criteria.py. Regime-specific relaxations/tightenings live in
# threshold_logic.REGIME_THRESHOLDS (which layers over BASE_THRESHOLDS).
BASE_THRESHOLDS: dict[str, float] = {
    "roe_min": 0.15,
    "roic_min": 0.15,
    "de_max": 1.0,
    "ev_max": 9.0,
    "pb_max": 1.5,
    "mca_max": 0.5,
}

# Named aliases for the older scripts that used module-level constants.
ROE_MIN = BASE_THRESHOLDS["roe_min"]
ROIC_MIN = BASE_THRESHOLDS["roic_min"]
DE_MAX = BASE_THRESHOLDS["de_max"]
EV_MAX = BASE_THRESHOLDS["ev_max"]
PB_MAX = BASE_THRESHOLDS["pb_max"]
MCA_MAX = BASE_THRESHOLDS["mca_max"]

# ── Canonical quality/value composite (preferred_metrics family) ────────────
# The same weighted formula appears in fundamentals_history.py and
# inclusion_criteria.py; keep it in ONE place so weights can't drift.
# q (quality) = w_qroe*clip(roe/0.25) + w_qroic*clip(roic/0.25)
#             + w_qde*clip(1 - de/2)  + w_qstab*earnings_stability
# v (value)   = w_vev*inv(ev) + w_vpb*inv(pb) + w_vmca*inv(mca)
# composite   = w_q*q + w_v*v
Q_W_ROE, Q_W_ROIC = 0.35, 0.35
Q_W_DE, Q_W_STAB = 0.15, 0.15
V_W_EV, V_W_PB, V_W_MCA = 0.4, 0.3, 0.3
COMP_W_Q, COMP_W_V = 0.55, 0.45
Q_SCALE = 0.25  # roe/roic denominator


def quality_value_composite(roe=None, roic=None, de=None, earnings_stability=None,
                            ev=None, pb=None, mca=None,
                            ev_max=EV_MAX, pb_max=PB_MAX, mca_max=MCA_MAX) -> float:
    """Weighted quality+value composite (0..1) — single canonical formula."""
    q, v = quality_value_parts(
        roe=roe, roic=roic, de=de, earnings_stability=earnings_stability,
        ev=ev, pb=pb, mca=mca, ev_max=ev_max, pb_max=pb_max, mca_max=mca_max,
    )
    return float(COMP_W_Q * q + COMP_W_V * v)


def quality_value_parts(roe=None, roic=None, de=None, earnings_stability=None,
                        ev=None, pb=None, mca=None,
                        ev_max=EV_MAX, pb_max=PB_MAX, mca_max=MCA_MAX) -> tuple[float, float]:
    """(quality_score, value_score) components of the canonical composite."""
    def _clip(x):
        return float(np.clip(x, 0.0, 1.0))
    def _inv(val, thr):
        if val is None or pd.isna(val):
            return 0.0
        return 1.0 if val <= thr else _clip(1 - (val - thr) / (thr * 1.5))
    q = 0.0
    if roe is not None and pd.notna(roe):
        q += Q_W_ROE * _clip(roe / Q_SCALE)
    if roic is not None and pd.notna(roic):
        q += Q_W_ROIC * _clip(roic / Q_SCALE)
    if de is not None and pd.notna(de):
        q += Q_W_DE * _clip(1 - de / 2)
    if earnings_stability is not None and pd.notna(earnings_stability):
        q += Q_W_STAB * float(earnings_stability)
    v = V_W_EV * _inv(ev, ev_max) + V_W_PB * _inv(pb, pb_max) + V_W_MCA * _inv(mca, mca_max)
    return float(q), float(v)


def prices_path(prefer_clean: bool = True) -> Path:
    clean = DATA_DIR / "daily_prices_clean.parquet"
    raw = DATA_DIR / "daily_prices.parquet"
    if prefer_clean and clean.exists():
        return clean
    return raw


def load_prices_pandas(prefer_clean: bool = True, tickers: Optional[list[str]] = None) -> pd.DataFrame:
    path = prices_path(prefer_clean)
    if HAS_POLARS:
        lf = pl.scan_parquet(str(path)).with_columns(
            pl.col("date").cast(pl.Date, strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
        )
        if tickers:
            lf = lf.filter(pl.col("ticker").is_in([t.upper() for t in tickers]))
        df = lf.collect().to_pandas()
    else:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        if tickers:
            df = df[df["ticker"].isin([t.upper() for t in tickers])]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


def load_adj_prices_pandas(tickers: Optional[list[str]] = None) -> pd.DataFrame:
    """Split/dividend-adjusted closes from daily_prices.parquet (adj_close).

    Return math for long-horizon backtests must use adj_close — raw ``close``
    carries split artifacts (a 4:1 split looks like a -75% day). Prefer this
    over load_prices_pandas for any engine that computes returns.
    """
    path = DATA_DIR / "daily_prices.parquet"
    if HAS_POLARS:
        lf = pl.scan_parquet(str(path)).select(
            pl.col("date").cast(pl.Date, strict=False),
            pl.col("ticker"),
            pl.col("adj_close").cast(pl.Float64, strict=False),
        )
        if tickers:
            lf = lf.filter(pl.col("ticker").is_in([t.upper() for t in tickers]))
        df = lf.collect().to_pandas()
    else:
        df = pd.read_parquet(path, columns=cols)
        if tickers:
            df = df[df["ticker"].isin([t.upper() for t in tickers])]
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"adj_close": "close"})
    return df.sort_values(["ticker", "date"])


def wide_closes(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()


def to_date_keys(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Cast daily date-key columns to python ``datetime.date`` so the parquet
    writer serializes them as DATE (no time component), not TIMESTAMP.

    Daily date-keys (trade dates, as-of dates, snapshot dates) should be DATE,
    not midnight-stamped TIMESTAMP. Intraday event times (fills, alerts,
    last_updated) must NOT be passed here.
    """
    df = df.copy()
    if isinstance(cols, str):
        cols = [cols]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df


def simple_returns(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.pct_change()


def clip_returns(rets: pd.DataFrame, clip: float = 0.35) -> pd.DataFrame:
    if clip and clip > 0:
        return rets.clip(lower=-clip, upper=clip)
    return rets


def load_membership() -> pd.DataFrame:
    p = DATA_DIR / "monitored_stocks.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def load_preferred() -> pd.DataFrame:
    p = DATA_DIR / "preferred_metrics.parquet"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def ann_stats(rets: pd.Series, rf: float = 0.04) -> dict:
    r = rets.dropna()
    if len(r) < 5:
        return {}
    ann_ret = float((1 + r.mean()) ** 252 - 1) if abs(r.mean()) < 0.5 else float(r.mean() * 252)
    # use compound for total path if levels available — mean*252 is ok for clipped daily
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "n": int(len(r))}
