"""vol_steps.py - Volatility-based per-ticker training-step allocator.

Rationale (from pass-1/2 sweeps):
  * Low-vol AEP plateaus ~9k steps; high-vol NVR still improving at 12k.
  * So step budget scales WITH volatility (more volatile => harder to fit =>
    more steps).
  * Steps also need to scale with window count (epochs) or fuller data
    underfits (pass-1 daily_stride1 was worse than fixed200 at equal steps).

Design (final):
  ann_vol      = trailing-window std of daily log-returns * sqrt(252)
  vol_ratio    = ann_vol / median_ann_vol_over_universe   (=> 1.0 at median)
  epoch_factor = sqrt(n_windows / REFERENCE_WINDOWS)       (gentle; full
                 multiplication overshoots the ceil for deep histories)
  steps = clamp(round(BASE * vol_ratio * epoch_factor), FLOOR, CEIL)
"""
import numpy as np
import pandas as pd
import granite_backfill as b
from granite_backfill import _clean_price_frame


def annualized_vol(series: np.ndarray, window: int = 1260) -> float:
    """Annualized vol from daily log-returns over a TRAILING window.

    Prices <= 1.0 are dropped (adj_close can be tiny far in the past; log
    returns off near-zero prices are meaningless).
    """
    s = pd.Series(np.asarray(series, dtype=float).ravel()).dropna()
    s = s[s > 1.0]
    if len(s) < 10:
        return float("nan")
    r = np.log(s.to_numpy()[1:] / s.to_numpy()[:-1])
    r = r[np.isfinite(r)]
    if len(r) > window:
        r = r[-window:]
    if len(r) < 3:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(252))


def allocate_steps(vols: dict, n_windows: dict | None = None,
                   base: int = 9000, median_vol: float | None = None,
                   ref_windows: int = 200, floor: int = 2000, ceil: int = 20000,
                   epoch_scale: bool = True) -> dict:
    vals = np.array([v for v in vols.values() if np.isfinite(v)])
    med = median_vol if median_vol is not None else float(np.median(vals))
    out = {}
    for tk, v in vols.items():
        if not np.isfinite(v) or med <= 0:
            out[tk] = base
            continue
        steps = base * (v / med)
        if epoch_scale and n_windows and tk in n_windows and n_windows[tk] > 0:
            steps *= (n_windows[tk] / ref_windows) ** 0.5
        out[tk] = int(np.clip(round(steps), floor, ceil))
    return out


def universe_vol_table() -> pd.DataFrame:
    """Compute ann_vol + recommended steps for the whole universe."""
    RAW = pd.read_parquet(b.PRICES)
    clean = _clean_price_frame(RAW, recent_trading_days=None)
    vols, nw = {}, {}
    for tk, g in clean.groupby("ticker"):
        s = g["close"].to_numpy().astype(float).ravel()
        s = s[np.isfinite(s)]
        if len(s) < (b.gd.CONTEXT + b.gd.HORIZON):
            continue
        vols[tk] = annualized_vol(s)
        nw[tk] = max(0, len(s) - (b.gd.CONTEXT + b.gd.HORIZON) + 1)
    steps = allocate_steps(vols, nw)
    rows = [{"ticker": tk, "ann_vol": vols[tk], "windows": nw[tk], "steps": steps[tk]}
            for tk in vols]
    return pd.DataFrame(rows).sort_values("ann_vol")


if __name__ == "__main__":
    df = universe_vol_table()
    print(f"universe tickers with enough history: {len(df)}")
    print(f"median ann_vol: {df['ann_vol'].median():.3f}")
    print(f"steps: min={df['steps'].min()} median={int(df['steps'].median())} max={df['steps'].max()}")
    print("\nLowest-vol (calmest):")
    print(df.head(8).to_string(index=False))
    print("\nHighest-vol:")
    print(df.tail(8).to_string(index=False))
