#!/usr/bin/env python3
"""shock_ride.py — ride basket/ticker price explosions, exit before crisis.

Uses DYNAMIC baskets from macro_sector_shock (GICS sectors + sub-industries
+ factor_groups) AND a per-ticker ride rule over the full price universe.

Rule (per basket or per ticker, monthly, no lookahead):
  ENTER  when 12m mom > entry_thresh (default 0.40) AND 3m mom > 0
  EXIT   when 3m mom <= 0
  position shifts 1 month after signals

Outputs:
  shock_ride.parquet — basket ride stats (dynamic baskets)
  shock_ride_tickers.parquet — per-ticker ride stats + CURRENT position:
      ticker, name, sector, n_trades, in_market_share, buy_hold_return,
      ride_return, excess, max_dd_ride, max_dd_buyhold, mom1, mom3, mom12,
      ride_long (current), recommendation, interpretation
Usage: python shock_ride.py [--save] [--entry 0.40]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from macro_sector_shock import _build_baskets, _load_price_matrix, _monthly_returns, _price_universe
from momentum_research import research_report
from breakout_detector import fresh_breakout_score
from fractal_windows import (fractal_signal_vec, fractal_consensus, best_span_wins,
                             fractal_multi_view, fractal_posture, momentum_stack)
from ride_longevity import (long_ride_score, ride_gate, ride_exit,
                            structural_gate, structural_positions,
                            STRUCTURAL_MODES)

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "shock_ride.parquet"
OUT_TICKERS = DATA_DIR / "shock_ride_tickers.parquet"
MIN_TICKER_HISTORY = 36  # months of price history required for a ticker ride
MAX_TICKERS = 600        # cap for the per-ticker pass (universe is ~583)


def max_dd(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak - 1).min()
    return float(dd)


def _ride_stats(m: pd.Series, entry_thresh: float) -> dict:
    """Classic ride: enter when 12m mom > entry_thresh, exit when 3m mom <= 0."""
    if len(m) < 12:
        return {}
    mom12 = m.rolling(12).sum()
    mom3 = m.rolling(3).sum()
    mom1 = m.rolling(1).sum()
    pos = (mom12 > entry_thresh).astype(int) & (mom3 > 0).astype(int)
    pos = pos.shift(1).fillna(0)
    ride = (m * pos).sum()
    bh = m.sum()
    return {
        "n_trades": int((pos.diff().fillna(0) != 0).sum()),
        "in_market_share": round(float(pos.mean()), 4),
        "buy_hold_return": round(float(bh), 4),
        "ride_return": round(float(ride), 4),
        "excess": round(float(ride - bh), 4),
        "max_dd_ride": round(max_dd((1 + m * pos).cumsum()), 4),
        "max_dd_buyhold": round(max_dd((1 + m).cumsum()), 4),
        "mom1": round(float(mom1.iloc[-1]), 4) if len(mom1) else np.nan,
        "mom3": round(float(mom3.iloc[-1]), 4) if len(mom3) else np.nan,
        "mom12": round(float(mom12.iloc[-1]), 4) if len(mom12) else np.nan,
        "ride_long": int(pos.iloc[-1]) if len(pos) else 0,
        "as_of": m.index[-1].strftime("%Y-%m-%d") if len(m) else "",
    }


def run(entry_thresh: float = 0.40, save: bool = True) -> int:
    # ── basket ride ──
    have = _price_universe()
    baskets = _build_baskets(have)
    rows = []
    print(f"=== shock ride — dynamic baskets (entry: 12m mom > {entry_thresh:.0%}, exit: 3m mom <= 0) ===")
    print(f"  dynamic baskets: {len(baskets)}")
    for bid, cfg in sorted(baskets.items()):
        m = _monthly_returns(cfg["tickers"])
        if m.empty or len(m) < 24:
            continue
        st = _ride_stats(m, entry_thresh)
        rows.append({"basket": bid, "basket_kind": "sector", "label": bid, "n_members": len(cfg["tickers"]), **st})
    bdf = pd.DataFrame(rows)
    if len(bdf):
        print(f"  Baskets where ride beats buy-hold: {(bdf['excess'] > 0).sum()}/{len(bdf)}")
        print(f"  Mean excess: {bdf['excess'].mean()*100:.1f}% | mean maxDD ride {bdf['max_dd_ride'].mean()*100:.1f}% vs BH {bdf['max_dd_buyhold'].mean()*100:.1f}%")
    if save:
        bdf.to_parquet(OUT, index=False)
        print(f"Wrote {OUT}")

    # ── per-ticker ride ──
    print(f"\n=== per-ticker ride (universe {len(have)} tickers, min {MIN_TICKER_HISTORY}mo history) ===")
    w = _load_price_matrix()
    meta = None
    try:
        ms = pd.read_parquet(DATA_DIR / "monitored_stocks.parquet")
        meta = dict(zip(ms["ticker"].astype(str).str.upper(), ms["name"]))
        sec = dict(zip(ms["ticker"].astype(str).str.upper(), ms["sector"]))
    except Exception:
        pass
    # volume matrix for fresh-breakout confirmation (OBV / volume expansion)
    volmat = None
    try:
        vp = pd.read_parquet(DATA_DIR / "daily_prices.parquet",
                             columns=["date", "ticker", "volume"])
        volmat = vp.pivot(index="date", columns="ticker", values="volume")
    except Exception:
        pass

    trows = []
    tickers = sorted(t for t in have if t in w.columns)
    tickers = tickers[:MAX_TICKERS]
    # ADV proxy (mean |daily log ret| x mean close) as a liquidity filter input
    adv_proxy = w.abs().mean()  # rough turnover/liquidity proxy per ticker
    for t in tickers:
        s = w[t].dropna()
        if len(s) < 3 * 21:  # 3 months floor (Ritter: post-first-month)
            continue
        # fresh-breakout verdict (near-high, acceleration, volume) on DAILY data
        vol = volmat[t].dropna() if volmat is not None and t in volmat.columns else None
        fb = fresh_breakout_score(s, vol) if vol is not None else fresh_breakout_score(s, None)
        fb_last = fb.iloc[-1] if len(fb) else None
        # fractal multi-view: granularity ladder 15d/30d/45d/90d + stack + posture
        try:
            mv = fractal_multi_view(s, configs=[(5, 3), (10, 3), (15, 3), (30, 3)])
            fcons_90 = mv["30x3"]["consensus"]
            best_90 = mv["30x3"]["best"]
            fcons_30 = mv["10x3"]["consensus"]
            best_30 = mv["10x3"]["best"]
            frac_u_90 = float(fcons_90["frac_uptrend"].iloc[-1]) if len(fcons_90) else np.nan
            frac_u_30 = float(fcons_30["frac_uptrend"].iloc[-1]) if len(fcons_30) else np.nan
            best_confirmed_90 = int(best_90["confirmed"].iloc[-1]) if len(best_90) else 0
            best_confirmed_30 = int(best_30["confirmed"].iloc[-1]) if len(best_30) else 0
            best_span_90 = int(best_90["best_span_len"].iloc[-1]) if len(best_90) else 0
            best_span_30 = int(best_30["best_span_len"].iloc[-1]) if len(best_30) else 0
            fp = fractal_posture(mv)
            fposture = fp["posture"]
            fposture_trend = fp["trend"]
            fposture_freshness = fp["freshness"]
            ms = momentum_stack(mv)
            stack_depth = ms["stack_depth"]
            stack_full = ms["full_stack"]
            stack_mom = ms["stack_mom"]
            # 15d + 45d best-span confirmation for the ladder
            best_15 = mv["5x3"]["best"]
            best_45 = mv["15x3"]["best"]
            best_confirmed_15 = int(best_15["confirmed"].iloc[-1]) if len(best_15) else 0
            best_confirmed_45 = int(best_45["confirmed"].iloc[-1]) if len(best_45) else 0
        except Exception:
            frac_u_90 = frac_u_30 = np.nan
            best_confirmed_90 = best_confirmed_30 = 0
            best_span_90 = best_span_30 = 0
            fposture = "WEAK"
            fposture_trend = "flat"
            fposture_freshness = "steady"
            stack_depth = stack_full = stack_mom = 0
            best_confirmed_15 = best_confirmed_45 = 0

        # long-ride durability (smoothness, pullback, overshoot, volume acc)
        try:
            lr = long_ride_score(s, vol) if vol is not None else long_ride_score(s, None)
            long_ride_last = float(lr["long_ride_score"].iloc[-1]) if len(lr) else 0.0
        except Exception:
            long_ride_last = 0.0

        m = np.log(s / s.shift(1))
        m = m.replace([np.inf, -np.inf], np.nan).dropna()
        # capture as_of from daily data BEFORE monthly resample (m.index[-1] would be month-end)
        as_of_date = s.index[-1].strftime("%Y-%m-%d")
        m = m.resample("ME").sum().dropna()
        if len(m) < 3:
            continue
        ann_vol = m.tail(12).std() * np.sqrt(12) if len(m) >= 2 else np.nan
        # research report (all measures + young-gate), Ritter first-month drop inside
        rr = research_report(m, annual_vol=ann_vol,
                             adv=float(adv_proxy[t]) if t in adv_proxy.index else None,
                             adv_series=adv_proxy)
        yg = rr["young_gate"]
        established = len(m) >= MIN_TICKER_HISTORY
        if established:
            st = _ride_stats(m, entry_thresh)
        else:
            # young ticker: build a minimal _ride_stats-equivalent so the row is uniform
            cum = (1 + m).cumprod()
            st = {
                "n_trades": 0,
                "in_market_share": 0.0,
                "buy_hold_return": round(float(m.sum()), 4),
                "ride_return": 0.0,
                "excess": 0.0,
                "max_dd_ride": np.nan,
                "max_dd_buyhold": np.nan,
                "mom1": rr.get("mom_1m"),
                "mom3": rr.get("mom_3m_ann"),
                "mom12": rr.get("mom_6m_ann"),
                "ride_long": int(yg["gate_open"]),
                "as_of": as_of_date,
            }
        # recommendation: quality-based ride gate (no 12mo requirement) + dual exit
        bv = fb_last["verdict"] if fb_last is not None else "NO_SIGNAL"
        rg = ride_gate(m, entry_thresh=entry_thresh,
                       stack_depth=stack_depth, long_ride=long_ride_last,
                       reliability=yg["reliability"])
        # if currently long, test the confirmed-breakdown exit
        ex = ride_exit(m, stack_depth=stack_depth, long_ride=long_ride_last,
                       trailing_stop=-0.25)
        # second-generation structural gate (all modes) on the daily series
        try:
            stg = {md: structural_gate(s, mode=md) for md in STRUCTURAL_MODES}
        except Exception:
            stg = {md: {"gate_open": False, "signal": 0.0} for md in STRUCTURAL_MODES}
        hot = (st["mom12"] > 0.40) if established else (rg["mom_used"] > entry_thresh if pd.notna(rg["mom_used"]) else False)
        fresh = bv == "FRESH_BREAKOUT"
        build = bv == "BUILDING"
        # fractal confirmation: best span confirmed OR strong consensus in 90d/30d
        fractal_90_ok = (best_confirmed_90 == 1) or (frac_u_90 >= 0.6 if not np.isnan(frac_u_90) else False)
        fractal_30_ok = (best_confirmed_30 == 1) or (frac_u_30 >= 0.6 if not np.isnan(frac_u_30) else False)

        if not established:
            # young / short-history: quality gate opens on any horizon w/ strong stack
            if rg["gate_open"]:
                rec, interp = "BUY", (
                    f"quality gate OPEN ({rg['horizon']} mom {rg['mom_used']:+.0%} ann, "
                    f"stack {stack_depth}/4, durability {long_ride_last:.2f}, "
                    f"fractal {fposture}/{fposture_trend}). Early durable momentum — "
                    f"most gains ahead."
                )
            else:
                rec, interp = "FLAT", (
                    f"quality gate closed ({rg['horizon']}, mom {rg['mom_used']:+.0%} "
                    f"ann, stack {stack_depth}/4, dur {long_ride_last:.2f}): "
                    f"{', '.join(rg['reasons']) or 'no signal'}."
                )
        elif st["ride_long"] and ex["exit"]:
            rec, interp = "AVOID", (
                f"CONFIRMED ride-over ({ex['exit_kind']}: 3m {ex['mom3']:+.0%}, "
                f"stack {stack_depth}/4, dur {long_ride_last:.2f}) — exit. "
                f"{', '.join(ex['reasons'])}."
            )
        elif st["ride_long"] and hot and (fresh or build) and (fractal_90_ok or fractal_30_ok):
            tag = "FRESH" if fresh else "BUILDING"
            rec, interp = "BUY", (
                f"{tag} breakout, explosion accelerating (12m {st['mom12']:+.0%}, "
                f"3m {st['mom3']:+.0%}, 1m {st['mom1']:+.0%}, "
                f"fractal={fposture}/{fposture_trend}, stack={stack_depth}/4, "
                f"dur={long_ride_last:.2f}, 90_cons={frac_u_90:.0%} 30_cons={frac_u_30:.0%}, "
                f"fresh_score {fb_last['fresh_score']:.2f})."
            )
        elif st["ride_long"] and hot and (fractal_90_ok or fractal_30_ok):
            rec, interp = "BUY", (
                f"explosion still accelerating (12m {st['mom12']:+.0%}, "
                f"3m {st['mom3']:+.0%}, 1m {st['mom1']:+.0%}, "
                f"fractal={fposture}/{fposture_trend}, stack={stack_depth}/4, "
                f"dur={long_ride_last:.2f}, 90_cons={frac_u_90:.0%} 30_cons={frac_u_30:.0%}) "
                f"— but NOT fresh ({bv})."
            )
        elif bv == "EXHAUSTED" and hot:
            rec, interp = "AVOID", (
                f"EXHAUSTED breakout (12m {st['mom12']:+.0%}, near-high, "
                f"volume divergence) — buying the top, not fresh."
            )
        elif st["ride_long"] and hot:
            rec, interp = "STAND DOWN", (
                f"momentum says long (12m {st['mom12']:+.0%}, 3m {st['mom3']:+.0%}, "
                f"1m {st['mom1']:+.0%}) — 1m rolling over. Dual exit: hold while "
                f"stack holds (stack {stack_depth}/4), exit on confirmed breakdown."
            )
        elif st["mom12"] > 0.40 and st["mom3"] <= 0:
            rec, interp = "AVOID", (
                f"exploded (12m {st['mom12']:+.0%}) but 3m {st['mom3']:+.0%} "
                f"(1m {st['mom1']:+.0%}) — confirmed rollover; ride exited."
            )
        elif st["mom12"] > 0.40:
            rec, interp = "WATCH", (
                f"12m {st['mom12']:+.0%} — above threshold but 3m {st['mom3']:+.0%} "
                f"not yet positive; waiting for entry."
            )
        else:
            rec, interp = "FLAT", f"12m {st['mom12']:+.0%} / 3m {st['mom3']:+.0%} — no signal."
        trows.append({
            "ticker": t,
            "name": (meta or {}).get(t, ""),
            "sector": (sec or {}).get(t, ""),
            **st,
            "is_young": bool(not established),
            "tsmom_3mo_sharpe": rr["tsmom_3mo_sharpe"],
            "tsmom_6mo_sharpe": rr["tsmom_6mo_sharpe"],
            "tsmom_12mo_sharpe": rr["tsmom_12mo_sharpe"],
            "stmom_1m_ret": rr["stmom_1m_ret"],
            "gw_high_prox": rr["gw52_high_prox"],
            "young_gate_open": yg["gate_open"],
            "young_gate_reliability": yg["reliability"],
            "fresh_verdict": bv,
            "fresh_score": fb_last["fresh_score"] if fb_last is not None else np.nan,
            "fractal_90_consensus": frac_u_90,
            "fractal_30_consensus": frac_u_30,
            "fractal_90_best_confirmed": best_confirmed_90,
            "fractal_30_best_confirmed": best_confirmed_30,
            "fractal_15_best_confirmed": best_confirmed_15,
            "fractal_45_best_confirmed": best_confirmed_45,
            "fractal_90_best_span": best_span_90,
            "fractal_30_best_span": best_span_30,
            "fractal_posture": fposture,
            "fractal_posture_trend": fposture_trend,
            "fractal_posture_freshness": fposture_freshness,
            "fractal_stack_depth": stack_depth,
            "fractal_stack_full": int(stack_full),
            "fractal_stack_mom": stack_mom,
            "long_ride_score": long_ride_last,
            "ride_gate_open": rg["gate_open"],
            "ride_gate_horizon": rg["horizon"],
            "ride_gate_mom": rg["mom_used"],
            "ride_exit_flag": ex["exit"],
            "ride_exit_kind": ex["exit_kind"],
            "structural_mode": "hybrid",
            "structural_signal": stg.get("hybrid", {}).get("signal", 0.0),
            "structural_gate_open": int(stg.get("hybrid", {}).get("gate_open", False)),
            "structural_in_market": stg.get("hybrid", {}).get("in_market_fraction", 0.0),
            "structural_turtle": stg.get("turtle", {}).get("signal", 0.0),
            "structural_volscale": stg.get("volscale", {}).get("signal", 0.0),
            "structural_regime": stg.get("regime", {}).get("signal", 0.0),
            "structural_recouple": stg.get("recouple", {}).get("signal", 0.0),
            "structural_consensus": stg.get("consensus", {}).get("signal", 0.0),
            "recommendation": rec,
            "interpretation": interp,
        })

    tdf = pd.DataFrame(trows)
    if len(tdf):
        print(f"  Tickers where ride beats buy-hold: {(tdf['excess'] > 0).sum()}/{len(tdf)}")
        print(f"  Mean excess: {tdf['excess'].mean()*100:.1f}% | mean maxDD ride {tdf['max_dd_ride'].mean()*100:.1f}% vs BH {tdf['max_dd_buyhold'].mean()*100:.1f}%")
        print(f"  Recommendations: {tdf['recommendation'].value_counts().to_dict()}")
        print("\nTop 10 by excess:")
        top = tdf.nlargest(10, "excess")[["ticker", "excess", "ride_return", "buy_hold_return"]]
        for _, row in top.iterrows():
            print(f"  {row['ticker']:6s} excess {row['excess']*100:+.1f}%  ride {row['ride_return']*100:+.1f}% BH {row['buy_hold_return']*100:+.1f}%")
    if save:
        tdf.to_parquet(OUT_TICKERS, index=False)
        print(f"Wrote {OUT_TICKERS}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=float, default=0.40, help="12m momentum entry threshold")
    ap.add_argument("--save", action="store_true", default=True, help="write outputs")
    args = ap.parse_args()
    exit(run(args.entry, args.save))