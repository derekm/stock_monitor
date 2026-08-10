#!/usr/bin/env python3
"""
regime_ab_test.py — A/B test: market-level HMM regimes vs per-ticker sector regimes

Compares:
  A) Market regime (current pass8): tag_windows(all_wins, market_regime_s, dates)
  B) Per-ticker sector regime: tag_windows(all_wins, ticker_sector_regime_s, dates)

Measures: OOS directional accuracy per regime bucket
"""
from __future__ import annotations
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pass6 import (
    tag_windows, temporal_split, train_regime_model,
    MIN_TEST, GAP_DAYS, P2_WIN, REGIMES
)
from pass5 import persistence_on_test
from regime_forecast import clean_series_dated, load_regime_map, windows_with_dates

DATA_DIR = Path(__file__).parent


def build_ticker_regime_map(ticker: str) -> pd.Series:
    """Build per-ticker regime series from sector/subindustry baskets.
    Uses the ticker's GICS sector basket regime (primary fallback).
    """
    bm = pd.read_csv(DATA_DIR / "basket_members.csv")
    sr = pd.read_csv("subindustry_regime.csv", usecols=["basket", "date", "regime"])

    # Find the ticker's GICS sector basket
    rows = bm[(bm["ticker"] == ticker) & (bm["basket_kind"] == "gics_sector")]
    if rows.empty:
        return pd.Series(dtype=object)
    sector_basket = rows.iloc[0]["basket"]

    sub = sr[sr["basket"] == sector_basket].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.drop_duplicates(subset="date", keep="last").sort_values("date")
    return sub.set_index("date")["regime"]


def tag_windows_per_ticker(ticker: str, all_wins, dates, ticker_regime_s):
    """Tag windows using the ticker's own regime series (like tag_windows but per-ticker)."""
    from pass6 import tag_windows as tag_windows_market
    # Just reuse the same logic with a different regime_s
    out = []
    for c, t, k in all_wins:
        fpt = k + 512 - 1  # CONTEXT - 1
        if fpt < 0 or fpt >= len(dates):
            continue
        d = pd.Timestamp(dates[fpt])
        prior = ticker_regime_s[ticker_regime_s.index <= d]
        if not len(prior):
            continue
        reg = str(prior.iloc[-1])
        out.append((c, t, fpt, reg))
    return out


def run_ab_test(tickers: list[str], steps_list: list[int] = [3000, 6000],
                caps_list: list[int] = [100, 200], lrs: list[float] = [1e-4],
                head_only: bool = True, exog: bool = True) -> pd.DataFrame:
    """Run both regime approaches on the given tickers and collect OOS results."""
    market_regime_s = load_regime_map()
    market_regime_s.index = pd.to_datetime(market_regime_s.index)

    results = []

    for tk in tickers:
        s, dates = clean_series_dated(tk)
        n = len(s)
        boundary = int(n * 0.7)  # pass6 default split_frac

        all_wins = []
        from pass6 import P2_WIN as P2_WIN_LOCAL
        for wname, wp in P2_WIN_LOCAL.items():
            all_wins += windows_with_dates(s, 0, n, wp["stride"], wp["cap"], dates)

        # --- A) Market regime ---
        tagged_market = tag_windows(all_wins, market_regime_s, dates)
        by_regime_m = {r: [] for r in ["low_vol", "normal", "high_vol_stress"]}
        for w in tagged_market:
            reg = w[3]
            if reg in by_regime_m:
                by_regime_m[reg].append(w)

        # --- B) Per-ticker sector regime ---
        ticker_regime_s = build_ticker_regime_map(ticker)
        ticker_regime_s.index = pd.to_datetime(ticker_regime_s.index)
        tagged_ticker = tag_windows_per_ticker(ticker, all_wins, dates, ticker_regime_s)
        by_regime_t = {r: [] for r in ["low_vol", "normal", "high_vol_stress"]}
        for w in tagged_ticker:
            reg = w[3]
            if reg in by_regime_t:
                by_regime_t[reg].append(w)

        # Evaluate each regime for both approaches
        for reg in ["low_vol", "normal", "high_vol_stress"]:
            rw_m = by_regime_m.get(reg, [])
            rw_t = by_regime_t.get(reg, [])

            if len(rw_m) < 3 or len(rw_t) < 3:
                continue

            # Same temporal split boundary for fair comparison
            boundary = int(len(rw_m) * 0.7)  # crude approximation, real boundary used in train
            train_m, test_m = temporal_split(rw_m, boundary)
            train_t, test_t = temporal_split(rw_t, boundary)

            if len(test_m) < 3 or len(test_t) < 3:
                continue

            pers_m = persistence_on_test([(c, t) for c, t, *_ in test_m])
            pers_t = persistence_on_test([(c, t) for c, t, *_ in test_t])

            # Run a quick test with first config only (to keep runtime reasonable)
            for steps in steps_list[:1]:
                for cap in caps_list[:1]:
                    for lr in lrs[:1]:
                        for approach, rw_train, rw_test, pers in [
                            ("market", train_m, test_m, pers_m),
                            ("per_ticker", train_t, test_t, pers_t)
                        ]:
                            tag = f"AB|{tk}|{reg}|{approach}|st={steps}|cap={cap}|lr={lr}"
                            res = train_regime_model(
                                train_wins=rw_train,
                                test_wins=rw_test,
                                steps=steps,
                                tag=tag,
                                lr=1e-4,
                                head_only=True,
                                exog=False,
                            )
                            if res.get("skipped"):
                                continue
                            results.append({
                                "ticker": tk,
                                "regime": reg,
                                "approach": approach,
                                "steps": steps,
                                "cap": 100,
                                "lr": 1e-4,
                                "dir_acc": res.get("dir_acc"),
                                "mape": res.get("mape"),
                                "pers_dir": pers["dir_acc"] if pers else None,
                                "excess": (res.get("dir_acc", 0) - (pers["dir_acc"] if pers else 0)),
                                "n_train": len(rw_train),
                                "n_test": len(rw_test),
                            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    test_tickers = ["AEP", "NVR", "ABBV", "MOS", "BLK"]  # diverse sectors
    print(f"Running A/B test on {len(test_tickers)} tickers...")
    df = run_ab_test(test_tickers)
    print(df.to_string(index=False))

    if not df.empty:
        # Summary by approach
        summary = df.groupby("approach").agg(
            n=("excess", "count"),
            mean_excess=("excess", "mean"),
            win_rate=("excess", lambda x: (x > 0).mean()),
            mean_dir=("dir_acc", "mean"),
        )
        print("\n=== Approach comparison ===")
        print(summary.to_string())

        # Per-ticker, per-regime
        detail = df.groupby(["ticker", "regime", "approach"]).agg(
            excess=("excess", "mean"),
            dir_acc=("dir_acc", "mean"),
        ).reset_index()
        print("\n=== Per ticker/regime ===")
        print(detail.to_string(index=False))