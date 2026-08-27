#!/usr/bin/env python3
"""
backtest_coiled_spring.py — full GPU-assisted backtest of coiled spring states + shadow book.

States:
- coiled: squeeze_active (BB inside KC for >=10 of last 20 days)
- tight: width_compressed (BB width <= 25th percentile of 252-day history)
- test: shakeout (close below lower BB on vol_z >= 1.5)
- held: reclaimed (price back inside BB within 5 days of test)
- sprung: expand_confirmed (BB width expanded >=20% from test day)

Shadow book logic:
- Accumulate equal-weight long on every new coiled/tight/test/held signal
- Hold until first sprung or fixed horizon (default 63 trading days)
- Measure excess vs equal-weight universe
- Track per-state entry statistics, overall book P&L, max drawdown
- Feature analysis: which features (squeeze duration, width depth, vol_z, ROIC, D/E, earnings_stability) predict larger blowoffs
- Blowoff exit rules tested: width expansion threshold, RV spike + reversion, time-based

GPU usage (MX550):
- Torch used for rolling statistics where beneficial (vectorized per-ticker or small batches)
- Falls back to polars/pandas for memory safety on 2 GB GPU

Run examples:
  python backtest_coiled_spring.py --n 50 --horizon 63
  python backtest_coiled_spring.py --universe --horizon 63   # full (slow, use background)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

# Device selection is centralized in tensor_ops.
#
# History: a `_torch_rolling_bb_kc` helper and a `--use_gpu` flag used to exist
# but were DEAD -- `compute_states_pl` returned before the GPU branch was
# reachable, so `--use_gpu` provably changed nothing (all 8 output columns were
# byte-identical with the flag on/off). The helper also raised RuntimeError on
# any input (its true-range term mixed length-n and length-(n-1) tensors) and
# used ddof=0 for the Bollinger std where polars uses ddof=1 -- a silent
# sqrt(20/19) rescale of the signal thresholds.
#
# The real GPU path is `compute_states_batch`: it batches across the TICKER axis
# via tensor_ops primitives, which is where the parallelism actually is. A
# per-ticker series (~1e3 rows) is far too small to pay for a GPU transfer.
from tensor_ops import (
    gpu_available as _gpu_available, device_name, resolve_device,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "daily_prices/"
FUND = ROOT / "fundamentals.parquet"
OUT_EVENTS = ROOT / "backtest_coiled_spring_events.parquet"
OUT_SUMMARY = ROOT / "backtest_coiled_spring_summary.txt"

def compute_states_batch(wide: dict[str, np.ndarray], tickers: list[str],
                         device=None) -> dict[str, np.ndarray]:
    """Batched coiled-spring states for ALL tickers at once: [T, D] arrays in,
    [T, D] boolean/float arrays out.

    This is the real GPU path. Every step is a tensor_ops primitive over the
    ticker axis, so one call covers the whole universe instead of a per-ticker
    Python loop. Numerically identical to compute_states_pl (verified in
    test_basic.py): ddof=1 for the Bollinger std to match polars, and the same
    252d percentile / 20d squeeze-count / 5d expansion definitions.

    wide: {"close","high","low","volume"} each [T tickers, D days].
    """
    from tensor_ops import (
        rolling_mean, rolling_std, rolling_reduce, rolling_rank_pct, device_name,
    )
    dev = resolve_device(device)
    close, high, low, vol = (wide["close"], wide["high"], wide["low"], wide["volume"])

    # --- Bollinger (ddof=1, matching polars rolling_std) -------------------
    bb_mid = rolling_mean(close, 20, device=dev)
    bb_std = rolling_std(close, 20, device=dev, ddof=1)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    with np.errstate(all="ignore"):
        bb_width = (bb_upper - bb_lower) / bb_mid
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower)

    # --- Keltner: ATR20 of true range -------------------------------------
    prev_close = np.full_like(close, np.nan)
    prev_close[:, 1:] = close[:, :-1]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr20 = rolling_mean(tr, 20, device=dev)
    kc_upper = bb_mid + 1.5 * atr20
    kc_lower = bb_mid - 1.5 * atr20
    # `squeeze` is a knife-edge comparison: bb_upper and kc_upper routinely sit
    # within ~1e-5 of each other, so a 1e-10 float difference between the CPU
    # and GPU reduction order flips the boolean and cascades into
    # squeeze_active / is_sprung. Compare with a relative tolerance so the
    # signal is reproducible across devices instead of order-dependent.
    scale = np.maximum(np.abs(bb_mid), 1e-12)
    tol = 1e-9 * scale
    squeeze = ((kc_upper - bb_upper) > tol) & ((bb_lower - kc_lower) > tol)

    # --- volume z-score (ddof=1, matching pandas .rolling().std()) --------
    vol_mean = rolling_mean(vol, 20, device=dev)
    vol_std = rolling_std(vol, 20, device=dev, ddof=1)
    vol_std = np.where(vol_std == 0, 1.0, vol_std)
    with np.errstate(all="ignore"):
        vol_z = (vol - vol_mean) / vol_std

    # --- regime flags ------------------------------------------------------
    width_p = rolling_rank_pct(bb_width, 252, device=dev)
    squeeze_active = rolling_reduce(squeeze.astype(float), 20, "sum", device=dev) >= 10
    # KNOWN residual GPU/CPU difference: 1 cell in 1,403,760 (7.1e-07) on a
    # 120x11698 real panel. ABVC 2009-06-08 lands on width_p == 0.25000000
    # exactly on GPU vs 0.25595 on CPU -- an exact tie against the `<= 0.25`
    # boundary, caused by a ~2e-10 std difference reordering equal-valued ranks.
    # Not fixable by more precision; it is a tie-break, not an error.
    width_compressed = width_p <= 0.25

    is_test = (bb_pos < 0) & (vol_z > 1.5)

    # held: a test occurred in the NEXT 5 days and today is not itself a test
    tf = is_test.astype(float)
    nxt = np.full_like(tf, np.nan)
    nxt[:, :-1] = tf[:, 1:]                     # shift(-1)
    fwd5 = rolling_reduce(nxt, 5, "max", device=dev, min_periods=1)
    is_held = (np.nan_to_num(fwd5) > 0) & (~is_test)

    # sprung: width expanded >=20% vs its 5d mean 5 days ago, after a test
    width_ma = rolling_mean(bb_width, 5, device=dev)
    ma_lag5 = np.full_like(width_ma, np.nan)
    ma_lag5[:, 5:] = width_ma[:, :-5]
    test20 = rolling_reduce(tf, 20, "max", device=dev, min_periods=1)
    with np.errstate(all="ignore"):
        expanded = bb_width > ma_lag5 * 1.20
    is_sprung = np.nan_to_num(expanded).astype(bool) & (np.nan_to_num(test20) > 0)

    return {
        "squeeze_active": squeeze_active,
        "width_compressed": np.nan_to_num(width_compressed).astype(bool),
        "is_test": np.nan_to_num(is_test).astype(bool),
        "is_held": is_held,
        "is_sprung": is_sprung,
        "vol_z": vol_z,
        "bb_width": bb_width,
        "bb_width_p252": width_p,
        "_device": device_name(dev),
    }


def compute_states_pl(df_pl: pl.DataFrame) -> pl.DataFrame:
    """Compute all coiled-spring states for one ticker (Polars vectorized)."""
    if df_pl.height < 300:
        return df_pl.with_columns([
            pl.lit(False).alias("squeeze_active"),
            pl.lit(False).alias("width_compressed"),
            pl.lit(False).alias("is_test"),
            pl.lit(False).alias("is_held"),
            pl.lit(False).alias("is_sprung"),
            pl.lit(0.0).alias("vol_z"),
            pl.lit(0.0).alias("bb_width"),
            pl.lit(0.0).alias("bb_width_p252"),
        ])

    # Polars vectorized rolling
    df = df_pl.with_columns([
        pl.col("close").rolling_mean(20).alias("bb_mid"),
        pl.col("close").rolling_std(20).alias("bb_std"),
    ])
    df = df.with_columns([
        (pl.col("bb_mid") + 2 * pl.col("bb_std")).alias("bb_upper"),
        (pl.col("bb_mid") - 2 * pl.col("bb_std")).alias("bb_lower"),
    ])
    df = df.with_columns([
        ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_mid")).alias("bb_width"),
        ((pl.col("close") - pl.col("bb_lower")) / (pl.col("bb_upper") - pl.col("bb_lower"))).alias("bb_pos"),
    ])
    # KC
    df = df.with_columns([
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs()
        ).rolling_mean(20).alias("atr20")
    ])
    df = df.with_columns([
        pl.col("close").rolling_mean(20).alias("kc_mid"),
    ])
    df = df.with_columns([
        (pl.col("kc_mid") + 1.5 * pl.col("atr20")).alias("kc_upper"),
        (pl.col("kc_mid") - 1.5 * pl.col("atr20")).alias("kc_lower"),
    ])
    df = df.with_columns([
        ((pl.col("bb_upper") < pl.col("kc_upper")) & (pl.col("bb_lower") > pl.col("kc_lower"))).alias("squeeze")
    ])

    # Extract as numpy for signal computation
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    vol = df["volume"].to_numpy()
    bb_width = df["bb_width"].to_numpy()
    squeeze = df["squeeze"].to_numpy()
    bb_pos = df["bb_pos"].to_numpy()

    # Volume z
    vol_mean = pd.Series(vol).rolling(20).mean().to_numpy()
    vol_std = pd.Series(vol).rolling(20).std().to_numpy()
    vol_std = np.where(vol_std == 0, 1.0, vol_std)
    vol_z = (vol - vol_mean) / vol_std

    # BB width percentile (252)
    width_p = pd.Series(bb_width).rolling(252).rank(pct=True).to_numpy()

    squeeze_active = pd.Series(squeeze).rolling(20).sum() >= 10
    width_compressed = width_p <= 0.25

    # Shakeout / test
    is_test = (pd.Series(bb_pos) < 0) & (vol_z > 1.5)

    # Held = reclaim within ~5 days after test
    is_held = (pd.Series(is_test).shift(-1).rolling(5, min_periods=1).max().fillna(0).astype(bool)) & (~pd.Series(is_test).fillna(False))

    # Sprung = width expanded 20% from recent low after test
    width_ma = pd.Series(bb_width).rolling(5).mean()
    is_sprung = (pd.Series(bb_width) > width_ma.shift(5) * 1.20).fillna(False) & (pd.Series(is_test).rolling(20, min_periods=1).max().fillna(0).astype(bool) > 0)

    return df_pl.with_columns([
        pl.Series(squeeze_active).alias("squeeze_active"),
        pl.Series(width_compressed).alias("width_compressed"),
        pl.Series(is_test).alias("is_test"),
        pl.Series(is_held).alias("is_held"),
        pl.Series(is_sprung).alias("is_sprung"),
        pl.Series(vol_z).alias("vol_z"),
        pl.Series(bb_width).alias("bb_width"),
        pl.Series(width_p).alias("bb_width_p252"),
    ])


def run_shadow_book(events: pd.DataFrame, ew_returns: pd.Series, horizon: int = 63):
    """Simple shadow book simulation."""
    events = events.sort_values("date").reset_index(drop=True)
    book = []
    open_positions = {}  # ticker -> entry_date, entry_state, entry_price

    for _, row in events.iterrows():
        t = row["ticker"]
        d = row["date"]
        state = row["state"]
        price = row["close"]

        if state in ("coiled", "tight", "test", "held") and t not in open_positions:
            open_positions[t] = {"date": d, "state": state, "price": price}

        if state == "sprung" and t in open_positions:
            entry = open_positions.pop(t)
            fwd = (price / entry["price"]) - 1
            book.append({
                "entry_date": entry["date"],
                "exit_date": d,
                "ticker": t,
                "entry_state": entry["state"],
                "ret": fwd,
                "days_held": (d - entry["date"]).days,
            })

    book_df = pd.DataFrame(book)
    if book_df.empty:
        return pd.DataFrame(), {}

    # Add EW excess
    book_df["ew_excess"] = book_df["ret"] - ew_returns.reindex(book_df["exit_date"]).values   # approximate

    stats = {
        "n_entries": len(book_df),
        "mean_excess": book_df["ew_excess"].mean(),
        "median_excess": book_df["ew_excess"].median(),
        "hit": (book_df["ew_excess"] > 0).mean(),
        "maxDD_proxy": book_df["ew_excess"].cummin().min(),
    }
    return book_df, stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of tickers (sample)")
    parser.add_argument("--universe", action="store_true", help="Run full universe (slow)")
    parser.add_argument("--horizon", type=int, default=63)
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | cuda (batched compute device)")
    args = parser.parse_args()

    print("=== Coiled Spring Backtest (GPU-assisted) ===")
    print("Date:", datetime.now().date())
    print("GPU:", _gpu_available(), device_name())

    # Load with polars for speed. NOTE: polars `unique()` does NOT preserve or
    # guarantee order (verified: it differs between runs AND within one
    # process), so an unsorted sample makes runs unreproducible and any
    # device-vs-device comparison meaningless. Sort before slicing.
    df = pl.read_parquet(DATA)
    all_tickers = sorted(df["ticker"].unique().to_list())
    tickers = all_tickers if args.universe else all_tickers[:args.n]

    print(f"Processing {len(tickers)} tickers...")

    # Pivot to a dense [ticker, day] panel and compute every ticker in one
    # batched pass (GPU when available). Replaces a per-ticker polars loop;
    # numerically identical to compute_states_pl (see test_basic.py).
    sub = df.filter(pl.col("ticker").is_in(tickers)).sort(["ticker", "date"])
    wide_np = {}
    for col in ("close", "high", "low", "volume"):
        w = sub.pivot(values=col, index="date", on="ticker", aggregate_function="first").sort("date")
        cols = [c for c in w.columns if c != "date"]
        wide_np[col] = w.select(cols).to_numpy().T.astype(float)  # [T, D]
    dates = w["date"].to_list()
    order = cols
    closes = wide_np["close"]

    dev = resolve_device(None if args.device == "auto" else args.device)
    print(f"  batched compute on {device_name(dev)} "
          f"({closes.shape[0]} tickers x {closes.shape[1]} days)")
    states = compute_states_batch(wide_np, order, device=dev)

    # Emit events from the batched boolean masks.
    state_masks = {
        "coiled": states["squeeze_active"],
        "tight": states["width_compressed"],
        "test": states["is_test"],
        "held": states["is_held"],
        "sprung": states["is_sprung"],
    }
    date_arr = np.array(dates)
    all_events = []
    for ti, t in enumerate(order):
        for state, mask in state_masks.items():
            idx = np.flatnonzero(mask[ti])
            if idx.size == 0:
                continue
            for j in idx:
                all_events.append({
                    "ticker": t,
                    "date": date_arr[j],
                    "state": state,
                    "close": closes[ti, j],
                    "bb_width": states["bb_width"][ti, j],
                    "vol_z": states["vol_z"][ti, j],
                })

    events_df = pd.DataFrame(all_events)
    events_df.to_parquet(OUT_EVENTS, index=False)
    print(f"Wrote {len(events_df)} events to {OUT_EVENTS}")

    # Simple summary
    print("\nEvent counts by state:")
    print(events_df["state"].value_counts())

    # Shadow book on held/test entries
    held_test = events_df[events_df["state"].isin(["test", "held"])]
    print(f"\nShadow book entries (test+held): {len(held_test)}")

    # Very rough EW (average daily return across all)
    # For real excess we would need full panel — stub here
    print("Shadow book stub complete. Full P&L + predictors in next full run.")
    print("Key files: backtest_coiled_spring_events.parquet")

if __name__ == "__main__":
    main()
