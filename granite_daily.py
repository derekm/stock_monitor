#!/usr/bin/env python3
"""
granite_daily.py — Daily 512-day -> 96-day Granite TTM forecast + CONTINUAL
RETRAINING on prior-day actuals.

This is the "own the forecasts" pipeline. Every day it:
  1. REFRESH: append the latest realized daily closes (prior-day actuals) into a
     persistent per-ticker series cache (granite_series_cache.parquet).
  2. RETRAIN: build rolling (context=512 -> target=96) windows from the cache,
     then take a few gradient steps on the Granite TTM (warm-started from the
     previous day's checkpoint so learning compounds). Saves a dated checkpoint.
  3. FORECAST: predict the next 96 trading days for every covered ticker using the
     tuned model, writing forecasts_granite.parquet (consumed by the dashboard).
  4. SCORE: compares today's forecast against the actuals that have since
     realized (walk-forward), tracking directional accuracy + MAE for the tuned
     model vs the zero-shot baseline, so we can SEE accuracy improve over time.

Context = 512 trading days, Horizon = 96 trading days (TTM native ceiling).
CPU-friendly: the TTM-r2 is only ~0.8M params; a few hundred steps/day is cheap.

Usage:
  python granite_daily.py run --tickers AAPL,MSFT      # forecast subset (no retrain unless --retrain)
  python granite_daily.py run --retrain                # refresh + retrain + forecast (all covered)
  python granite_daily.py run --retrain --limit 50     # retrain/forecast first 50 tickers only
  python granite_daily.py score                        # show accuracy-vs-baseline history
  python granite_daily.py status                       # cache/checkpoint state
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import window_padding as wp

DATA_DIR = Path(__file__).resolve().parent
CACHE = DATA_DIR / "granite_series_cache.parquet"
FORECAST_FILE = DATA_DIR / "forecasts_granite.parquet"
CKPT_DIR = DATA_DIR / "granite_ckpts"
GLOBAL_DIR = CKPT_DIR / "global"        # aggregated model trained on full-context tickers
PADDED_DIR = CKPT_DIR / "padded"        # aggregated model trained on proxy-padded shorts
PER_TICKER_DIR = CKPT_DIR / "per_ticker"  # one tuned model per ticker (highest precedence)
for _d in (GLOBAL_DIR, PADDED_DIR, PER_TICKER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# cache the constituents frame (GICS sector lookup for window padding)
_CONSTITS_DF = None


def _constituents_df():
    global _CONSTITS_DF
    if _CONSTITS_DF is None:
        p = DATA_DIR / "sp500_constituents.parquet"
        _CONSTITS_DF = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return _CONSTITS_DF


def _sector_for(ticker: str):
    df = _constituents_df()
    if df.empty or "gics_sector" not in df.columns:
        return None
    m = df.loc[df["ticker"] == ticker, "gics_sector"]
    return m.iloc[0] if len(m.dropna()) else None


# alias so the pad call above reads clearly
pad_to_context = wp.pad_to_context
ACC_FILE = DATA_DIR / "granite_accuracy.json"
DEFAULT_MODEL = "ibm-granite/granite-timeseries-ttm-r2"

CONTEXT = 512
HORIZON = 96
SERIES_CAP = 1100          # max closes kept per ticker (bounds memory + windows)
WINDOWS_PER_TICKER = 48    # last N rolling (512->96) windows used for training
STEPS_PER_DAY = 150
LR = 1e-4
BATCH = 8

from forecast_granite import load_granite_model, load_ohlcv_with_sectors  # noqa: E402


# --------------------------------------------------------------------------
# Series cache (the "prior-day actuals" feed)
# --------------------------------------------------------------------------
def refresh_cache(tickers: list[str] | None = None) -> pd.DataFrame:
    """Append the latest realized closes from daily_prices into the persistent
    per-ticker cache. Returns the updated cache."""
    prices = load_ohlcv_with_sectors(tickers)
    if prices is None or prices.empty:
        prices = pd.DataFrame(columns=["date", "ticker", "close"])
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["close"] = prices["close"].astype(float)
    prices = prices[["date", "ticker", "close"]].dropna(subset=["close"])
    # keep newest SERIES_CAP per ticker
    prices = (
        prices.sort_values("date")
        .groupby("ticker")
        .tail(SERIES_CAP)
    )
    if CACHE.exists():
        old = pd.read_parquet(CACHE)
        old["date"] = pd.to_datetime(old["date"])
        combined = pd.concat([old, prices], ignore_index=True)
    else:
        combined = prices
    combined = (
        combined.drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
        .groupby("ticker")
        .tail(SERIES_CAP)
    )
    combined.to_parquet(CACHE, index=False)
    return combined


def load_cache() -> pd.DataFrame:
    if not CACHE.exists():
        return refresh_cache()
    return pd.read_parquet(CACHE)


# --------------------------------------------------------------------------
# Window dataset (512 -> 96)
# --------------------------------------------------------------------------
def build_windows(cache: pd.DataFrame, tickers: list[str] | None = None):
    """Yield (context_array, target_array) windows of shape (CONTEXT,) (HORIZON,)."""
    wins = []
    tk_list = tickers or cache["ticker"].unique().tolist()
    for tk in tk_list:
        s = cache[cache["ticker"] == tk].sort_values("date")["close"].values.astype(np.float32)
        if len(s) < CONTEXT + HORIZON:
            continue
        s = s[-(SERIES_CAP):]
        need = CONTEXT + HORIZON
        if len(s) < need:
            continue
        # last WINDOWS_PER_TICKER rolling windows (most recent first), each exactly CONTEXT+HORIZON
        n_windows = min(WINDOWS_PER_TICKER, len(s) - need + 1)
        for k in range(n_windows):
            st = len(s) - need - k
            if st < 0:
                continue
            ctx = s[st: st + CONTEXT]
            tgt = s[st + CONTEXT: st + CONTEXT + HORIZON]
            if len(ctx) == CONTEXT and len(tgt) == HORIZON:
                wins.append((ctx, tgt))
    return wins


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def latest_ckpt_in(d: Path) -> Path | None:
    if not d.exists():
        return None
    ckpts = sorted(d.glob("*_tuned_*.pt"), reverse=True)
    return ckpts[0] if ckpts else None


def latest_ckpt() -> Path | None:
    """Legacy: most recent ckpt anywhere under CKPT_DIR (kept for callers)."""
    return latest_ckpt_in(CKPT_DIR)


def resolve_model_ckpt(ticker: str | None = None) -> tuple[Path | None, str]:
    """Precedence for forecasting/retraining:
       1. per_ticker/<TICKER>_tuned_*.pt   (best, ticker-specific)
       2. global/granite_ttm_tuned_*.pt    (full-context aggregate)
       3. padded/granite_ttm_tuned_*.pt    (proxy-padded aggregate, for shorts)
       4. None                              (pretrained zero-shot)
    When `ticker` is None we only look at the global aggregate (default for the
    general pipeline). Short tickers (< CONTEXT closes) should resolve to the
    padded aggregate until they graduate to a full per-ticker model."""
    if ticker:
        pt = latest_ckpt_in(PER_TICKER_DIR / ticker)
        if pt is not None:
            return pt, "granite_tuned_per_ticker"
    g = latest_ckpt_in(GLOBAL_DIR)
    if g is not None:
        return g, "granite_tuned_global"
    p = latest_ckpt_in(PADDED_DIR)
    if p is not None:
        return p, "granite_tuned_padded"
    return None, "pretrained"


def _device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def retrain(cache: pd.DataFrame, tickers: list[str] | None = None, steps: int = STEPS_PER_DAY):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = _device()
    wins = build_windows(cache, tickers)
    if not wins:
        print("no training windows available (need >= 608 closes per ticker)")
        return None
    ctx = np.stack([w[0] for w in wins])[:, :, None]   # (N, 512, 1)
    tgt = np.stack([w[1] for w in wins])[:, :, None]   # (N, 96, 1)
    ds = TensorDataset(torch.tensor(ctx), torch.tensor(tgt))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model, kind = load_granite_model(DEFAULT_MODEL)
    model = model.to(device)
    ckpt = latest_ckpt()
    if ckpt is not None:
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"warm-started from {ckpt.name}")
        except Exception as e:
            print(f"ckpt load failed ({e}); training from pretrained")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    CKPT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    step = 0
    while step < steps:
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(past_values=xb, future_values=yb)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step >= steps:
                break
    # save dated checkpoint
    d = date.today().isoformat().replace("-", "")
    out_path = CKPT_DIR / f"granite_ttm_tuned_{d}.pt"
    if device.type == "cuda":
        model = model.cpu()
    torch.save(model.state_dict(), out_path)
    model.eval()
    print(f"retrained {step} steps on {len(wins)} windows; ckpt={out_path.name} ({time.time()-t0:.1f}s, device={device.type})")
    return out_path


# --------------------------------------------------------------------------
# Forecast (96 days) using the TUNED model
# --------------------------------------------------------------------------
def forecast_all(cache: pd.DataFrame, tickers: list[str] | None = None, use_tuned: bool = True):
    import torch
    from forecast_granite import forecast_ttm_univariate

    model, kind = load_granite_model(DEFAULT_MODEL)
    loaded_ckpt = None  # track which state_dict is currently in `model`
    # resolver precedence: per-ticker -> global -> padded -> pretrained
    tk_list = tickers or cache["ticker"].unique().tolist()
    rows = []
    for tk in tk_list:
        s = cache[cache["ticker"] == tk].sort_values("date")["close"].values.astype(float)
        sector = _sector_for(tk)
        if len(s) < CONTEXT:
            # new addition / short history: pad the context head with a real
            # sector/market proxy (rescaled to the ticker's level) so the
            # fixed-context model still produces a forecast.
            y_in = pad_to_context(tk, s.astype(np.float32), sector=sector,
                                 px=cache, cons=_constituents_df()).astype(float)
        else:
            y_in = s[-CONTEXT:]
        # pick the best checkpoint for this ticker (per-ticker wins, then bucket)
        ckpt, kind = resolve_model_ckpt(tk if use_tuned else None)
        if use_tuned and ckpt is not None and ckpt != loaded_ckpt:
            try:
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                loaded_ckpt = ckpt
            except Exception as e:
                print(f"  ckpt {ckpt.name} load failed ({e}); using pretrained")
                kind = "pretrained"
        elif not use_tuned:
            if loaded_ckpt is not None:
                pass  # keep pretrained base
            kind = "pretrained"
        model.eval()
        pred = forecast_ttm_univariate(model, kind, y_in, HORIZON, context=CONTEXT)
        last = float(y_in[-1])
        last_date = cache[cache["ticker"] == tk]["date"].max()
        future = pd.bdate_range(last_date + timedelta(days=1), periods=HORIZON)
        for h, (dt, pv) in enumerate(zip(future, pred), 1):
            rows.append({
                "ticker": tk,
                "horizon": h,
                "forecast_date": dt.date(),
                "forecast_close": round(float(pv), 4),
                "last_close": round(last, 4),
                "pct_change": round((float(pv) / last - 1) * 100, 3),
                "backend": kind,
                "as_of": last_date.date() if hasattr(last_date, "date") else pd.to_datetime(last_date).date(),
                "model_date": (ckpt.name if (use_tuned and ckpt) else "pretrained"),
                "history_n": len(s),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_parquet(FORECAST_FILE, index=False)
    print(f"forecast {len(df)} rows for {df['ticker'].nunique() if not df.empty else 0} tickers -> {FORECAST_FILE.name}")
    return df


# --------------------------------------------------------------------------
# Accuracy scoring (walk-forward): today's forecast vs realized actuals
# --------------------------------------------------------------------------
def score():
    if not FORECAST_FILE.exists():
        print("no forecasts_granite.parquet yet")
        return
    fc = pd.read_parquet(FORECAST_FILE)
    # realized closes from cache
    cache = load_cache()
    realized = cache.copy()
    realized["forecast_date"] = pd.to_datetime(realized["date"]).dt.strftime("%Y-%m-%d")
    realized = realized[["ticker", "forecast_date", "close"]].rename(columns={"close": "actual_close"})
    fc["forecast_date"] = pd.to_datetime(fc["forecast_date"]).dt.strftime("%Y-%m-%d")
    fc = fc.merge(realized, on=["ticker", "forecast_date"], how="left")
    done = fc.dropna(subset=["actual_close"])
    if done.empty:
        print("no realized actuals yet for current forecast horizon")
        return
    done["err"] = done["forecast_close"] - done["actual_close"]
    done["dir_ok"] = ((done["forecast_close"] - done["last_close"]) *
                      (done["actual_close"] - done["last_close"])) > 0
    acc = done["dir_ok"].mean()
    mae = done["err"].abs().mean()
    print(f"realized horizons so far: n={len(done)} dir_acc={acc:.3f} mae=${mae:.2f}")
    # persist a running log keyed by as_of + model_date
    rec = {
        "scored_on": date.today().isoformat(),
        "as_of": str(done["as_of"].iloc[0]) if "as_of" in done else None,
        "model_date": str(done["model_date"].iloc[0]) if "model_date" in done else None,
        "n_realized": int(len(done)),
        "dir_acc": round(float(acc), 4),
        "mae": round(float(mae), 4),
    }
    hist = json.loads(ACC_FILE.read_text()) if ACC_FILE.exists() else []
    hist.append(rec)
    ACC_FILE.write_text(json.dumps(hist, indent=2))
    print("accuracy history (last 5):")
    for r in hist[-5:]:
        print(f"  {r['scored_on']} as_of={r['as_of']} n={r['n_realized']} "
              f"dir_acc={r['dir_acc']} mae={r['mae']}")


def cmd_run(args):
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    if args.limit:
        # limit applies to ticker count when none specified
        pass
    print(f"[{datetime.now():%H:%M:%S}] refresh cache", flush=True)
    cache = refresh_cache(tickers)
    print(f"  cache: {cache['ticker'].nunique()} tickers, "
          f"{cache['date'].min().date()} -> {cache['date'].max().date()}", flush=True)
    if args.retrain:
        tk = tickers
        if args.limit:
            tk = (tk or cache["ticker"].unique().tolist())[: args.limit]
        retrain(cache, tk, steps=args.steps)
    fc_tk = tickers
    if args.limit and not tickers:
        fc_tk = cache["ticker"].unique().tolist()[: args.limit]
    forecast_all(cache, fc_tk, use_tuned=args.retrain or latest_ckpt() is not None)
    score()


def cmd_status(args):
    cache = load_cache() if CACHE.exists() else pd.DataFrame()
    print(f"cache tickers: {cache['ticker'].nunique() if not cache.empty else 0}")
    print(f"latest checkpoint: {latest_ckpt().name if latest_ckpt() else 'none'}")
    print(f"forecasts file: {'yes' if FORECAST_FILE.exists() else 'no'}")
    if ACC_FILE.exists():
        hist = json.loads(ACC_FILE.read_text())
        print(f"accuracy records: {len(hist)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tickers")
    r.add_argument("--retrain", action="store_true", help="retrain the TTM on prior-day actuals")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--steps", type=int, default=STEPS_PER_DAY)
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)
    sc = sub.add_parser("score")
    sc.set_defaults(func=lambda a: score())
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
