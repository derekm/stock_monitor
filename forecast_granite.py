#!/usr/bin/env python3
"""
forecast_granite.py — Stock forecasting with IBM Granite Time Series (TTM).

Aligned with Granite TTM best practices:
  - Consistent business-day OHLCV panels (via ttm_features)
  - Multivariate channels (price, volume, returns, vol, RSI, peers)
  - Per-channel scaling
  - Zero-shot forecast with optional iterative rolling extension
  - Directional accuracy + MAE/RMSE/MAPE evaluation
  - Portfolio / index / first-trade / days-ago (≤512) history windows

Install:
  pip install granite-tsfm transformers torch accelerate pyarrow

Usage:
  python forecast_granite.py forecast --index portfolio --from-first-trade --horizon 10
  python forecast_granite.py forecast --index portfolio,defensive,growth --horizon 10
  python forecast_granite.py forecast --index portfolio --index growth --days-ago 252
  python forecast_granite.py forecast --ticker MOS --days-ago 126 --horizon 10
  python forecast_granite.py forecast --ticker MOS,CF --multivariate --horizon 20
  python forecast_granite.py forecast --ticker MOS --channels full --horizon 15 --rolling
  python forecast_granite.py backtest --index portfolio,defensive --horizon 10 --window 40
  python forecast_granite.py status
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
SECTOR_PRICES_FILE = DATA_DIR / "sector_prices.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
FORECAST_FILE = DATA_DIR / "forecasts_granite.parquet"
FORECAST_CSV = DATA_DIR / "forecasts_granite.csv"
BACKTEST_FILE = DATA_DIR / "forecast_backtest_metrics.csv"

from granite_config import DEFAULT_MODEL  # canonical Granite model id

from ttm_features import (  # noqa: E402
    build_panel,
    build_multivariate_bundle,
    load_ohlcv,
)
from ttm_exogenous import build_exog_panel, merge_exog  # noqa: E402
from regime_serving import serve_regime_model, current_regime  # noqa: E402

def load_ohlcv_with_sectors(tickers: list[str] | None = None) -> pd.DataFrame:
    """Combine stock daily_prices with sector_prices for SECT_* tickers."""
    frames = []
    stock_tickers = [t for t in (tickers or []) if not str(t).startswith("SECT_")]
    sect_tickers = [t for t in (tickers or []) if str(t).startswith("SECT_")]
    if tickers is None or stock_tickers or not tickers:
        try:
            frames.append(load_ohlcv(stock_tickers if tickers else None))
        except Exception:
            pass
    if sect_tickers and SECTOR_PRICES_FILE.exists():
        sp = pd.read_parquet(SECTOR_PRICES_FILE)
        sp["date"] = pd.to_datetime(sp["date"])
        frames.append(sp[sp["ticker"].isin(sect_tickers)])
    elif tickers is None and SECTOR_PRICES_FILE.exists():
        sp = pd.read_parquet(SECTOR_PRICES_FILE)
        sp["date"] = pd.to_datetime(sp["date"])
        frames.append(sp)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "source"])
    return pd.concat(frames, ignore_index=True)



def _load_trades() -> Optional[pd.DataFrame]:
    for p in [
        DATA_DIR / "trades.parquet",
        DATA_DIR.parent / "trades.parquet",
        Path("/home/workdir/artifacts/trades.parquet"),
    ]:
        if p.exists():
            t = pd.read_parquet(p)
            if "filled_datetime" in t.columns:
                t["filled_datetime"] = pd.to_datetime(t["filled_datetime"])
            return t
    return None


def first_trade_dates() -> dict[str, pd.Timestamp]:
    trades = _load_trades()
    if trades is None or "filled_datetime" not in trades.columns:
        return {}
    buys = trades
    if "transaction_type" in trades.columns:
        buys = trades[trades["transaction_type"].isin(["Buy", "Dividend Reinvestment", "DRIP"])]
    g = buys.dropna(subset=["filled_datetime"]).groupby("ticker")["filled_datetime"].min()
    return {k: pd.Timestamp(v).normalize() for k, v in g.items()}


def portfolio_tickers() -> list[str]:
    hold = DATA_DIR / "portfolio_holdings.parquet"
    if hold.exists():
        h = pd.read_parquet(hold)
        if "ticker" in h.columns:
            return h["ticker"].astype(str).str.upper().tolist()
    if STOCKS_FILE.exists():
        s = pd.read_parquet(STOCKS_FILE)
        if "in_portfolio" in s.columns:
            return s[s["in_portfolio"] == True]["ticker"].tolist()
    trades = _load_trades()
    if trades is not None:
        return sorted(trades["ticker"].astype(str).str.upper().unique().tolist())
    return []


def sector_slugs() -> list[str]:
    meta = DATA_DIR / "sector_tickers.csv"
    if meta.exists():
        return pd.read_csv(meta)["ticker"].tolist()
    if SECTOR_PRICES_FILE.exists():
        return pd.read_parquet(SECTOR_PRICES_FILE)["ticker"].unique().tolist()
    return []


from cli_common import add_index_args, add_ticker_args, add_sector_arg, resolve_tickers_from_args, ticker_index_map_from_args
from index_registry import (  # noqa: E402
    available_indexes,
    parse_indexes,
    tickers_for_index as registry_tickers_for_index,
    ticker_index_map,
    index_help_text,
    canonicalize as canonicalize_index,
)


def parse_index_list(raw) -> list[str]:
    """Parse index args; supports 'all' via index_registry."""
    try:
        return parse_indexes(raw)
    except ValueError as e:
        raise SystemExit(str(e)) from e


def tickers_for_index(idx: str, stocks=None) -> list[str]:
    t = registry_tickers_for_index(idx)
    if idx in ("portfolio",) and not t:
        raise SystemExit("No portfolio tickers found")
    if idx == "sectors" and not t:
        raise SystemExit("No sector prices — run: python cross_asset_analysis.py save-sector-prices")
    return t


def resolve_ticker_index_map(args) -> dict[str, list[str]]:
    """Map ticker -> list of index names under the request."""
    if getattr(args, "ticker", None):
        mapping: dict[str, list[str]] = {}
        for t in args.ticker.split(","):
            t = t.strip().upper()
            if t:
                mapping.setdefault(t, [])
                if "custom" not in mapping[t]:
                    mapping[t].append("custom")
        return mapping

    if getattr(args, "sector", None):
        mapping = {}
        raw = [s.strip() for s in args.sector.split(",")]
        slug_map = {}
        meta = DATA_DIR / "sector_tickers.csv"
        if meta.exists():
            m = pd.read_csv(meta)
            slug_map = dict(zip(m["sector_name"].str.lower(), m["ticker"]))
            slug_map.update({x.lower(): x for x in m["ticker"]})
        for s in raw:
            key = s.lower()
            if key in slug_map:
                tk = slug_map[key]
            elif s.upper().startswith("SECT_"):
                tk = s.upper()
            else:
                tk = "SECT_" + "".join(ch if ch.isalnum() else "_" for ch in s).upper()[:24]
            mapping.setdefault(tk, [])
            if "sectors" not in mapping[tk]:
                mapping[tk].append("sectors")
        return mapping

    indexes = parse_index_list(getattr(args, "index", None))
    if indexes:
        return ticker_index_map(indexes)

    # default: fertilizer members if present
    default = ticker_index_map(["fertilizer"])
    if default:
        return default
    return {t: ["default"] for t in ["MOS", "CF", "SHEL"]}


def resolve_tickers(args) -> list[str]:
    return list(resolve_ticker_index_map(args).keys())


def index_label_for(ticker: str, mapping: dict[str, list[str]]) -> str:
    labels = mapping.get(ticker, [])
    return ",".join(labels) if labels else ""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_model = None
_model_name = None



def resolve_history_start(
    ticker: str,
    dates_index: pd.DatetimeIndex | None = None,
    *,
    days_ago: int | None = None,
    from_first_trade: bool = False,
    first_map: dict | None = None,
) -> pd.Timestamp | None:
    """Pick series start date.

    Priority:
      1. --days-ago N (1..512): last N observations on the available index
         (or calendar business days if index is empty)
      2. --from-first-trade: first filled trade for ticker
      3. None = full history
    """
    if days_ago is not None:
        n = int(days_ago)
        if n < 1:
            raise ValueError("--days-ago must be >= 1")
        if n > 512:
            raise ValueError("--days-ago must be <= 512 (TTM context ceiling)")
        if dates_index is not None and len(dates_index) > 0:
            idx = pd.DatetimeIndex(dates_index).sort_values()
            # take last n points → start at that timestamp
            if len(idx) <= n:
                return pd.Timestamp(idx[0])
            return pd.Timestamp(idx[-n])
        # no index yet — approximate with business days from today
        return pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(n - 1)

    if from_first_trade:
        first_map = first_map if first_map is not None else first_trade_dates()
        return first_map.get(ticker)
    return None


def load_granite_model(model_name: str = DEFAULT_MODEL):
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model, "granite"
    try:
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        import torch  # noqa: F401

        print(f"Loading Granite TTM: {model_name}")
        model = TinyTimeMixerForPrediction.from_pretrained(model_name)
        model.eval()
        _model, _model_name = model, model_name
        return model, "granite"
    except Exception as e1:
        try:
            from transformers import AutoModel
            import torch  # noqa: F401

            print(f"Loading via transformers: {model_name}")
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            model.eval()
            _model, _model_name = model, model_name
            return model, "transformers"
        except Exception as e2:
            print("Granite TTM unavailable — statistical fallback (drift + seasonal).")
            print(f"  ({type(e1).__name__}: {e1})")
            return None, "fallback"


def forecast_fallback(y: np.ndarray, horizon: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return np.full(horizon, y[-1] if len(y) else np.nan)
    drift = (y[-1] - y[0]) / max(len(y) - 1, 1)
    last = y[-1]
    season = min(5, len(y))
    out = []
    for h in range(1, horizon + 1):
        seas = y[-season + (h % season) - season]
        out.append(0.6 * (last + drift * h) + 0.4 * seas)
    return np.asarray(out, dtype=float)


def _channels_from_series(w) -> np.ndarray:
    """(close, pct_return, realized_vol20) — must match pass6._channels_from_close."""
    c = np.asarray(w, dtype=np.float32)
    r = np.zeros_like(c)
    r[1:] = np.diff(c) / np.clip(c[:-1], 1e-9, None)
    v = np.zeros_like(c)
    for i in range(len(c)):
        lo = max(0, i - 19)
        v[i] = np.std(r[lo:i + 1]) if i > lo else 0.0
    return np.stack([c, r, v], axis=-1)


def _event_proximity_series(n: int, end_ts: pd.Timestamp) -> np.ndarray:
    """Days-until-next-FOMC/expiry per timestep for the LAST n days ending at
    end_ts — the serving-side twin of pass6._exog_channel (known-future calendar
    events as an exogenous input channel; TTM paper 3.2, input form)."""
    out = np.zeros(n, dtype=np.float32)
    try:
        ev = pd.read_csv(Path(__file__).resolve().parent / "economic_calendar.csv")
        dates = []
        for _, r in ev.iterrows():
            et = str(r.get("event_type", ""))
            if "fomc" in et.lower() or "expiry" in et.lower() or "fed" in et.lower():
                try:
                    dates.append(pd.Timestamp(r["date"]))
                except Exception:
                    pass
        dates = sorted(dates)
        if not dates:
            return out
        day = pd.Timedelta(days=1)
        for i in range(n):
            t = end_ts - (n - 1 - i) * day
            nxt = min((d for d in dates if d >= t), default=None)
            if nxt is not None:
                out[i] = float(min((nxt - t).days, 180)) / 180.0
    except Exception:
        pass
    return out


def forecast_ttm_univariate(model, kind: str, y: np.ndarray, horizon: int, context: int = 512) -> np.ndarray:
    if model is None or kind == "fallback":
        return forecast_fallback(y, horizon)
    import torch

    y = np.asarray(y, dtype=np.float32)
    hist = y[-min(context, len(y)):]
    try:
        # multi-channel input: (context, n_ch) already built by the caller
        # (regime ensemble passes close+return+vol20+exog); else 1-channel.
        if hist.ndim == 1:
            x = torch.tensor(hist).view(1, -1, 1)
        else:
            x = torch.tensor(hist).unsqueeze(0)
        if next(model.parameters()).is_cuda:
            x = x.to(next(model.parameters()).device)
        fwd_kw = {}
        if getattr(model, "_rpt", False):
            # Resolution Prefix Tuning checkpoint: pass the daily freq token (2)
            fwd_kw["freq_token"] = torch.full((x.shape[0],), 2, dtype=torch.long,
                                              device=x.device)
        with torch.no_grad():
            out = model(past_values=x, **fwd_kw) if "past_values" in model.forward.__code__.co_varnames else model(x, **fwd_kw)
            if hasattr(out, "prediction_outputs"):
                pred = out.prediction_outputs
            elif isinstance(out, torch.Tensor):
                pred = out
            else:
                pred = out[0]
            pred = pred.detach().cpu().numpy()
            # (1, horizon, n_ch) -> (horizon,) on the close channel
            if pred.ndim == 3:
                pred = pred[0, :, 0]
            else:
                pred = pred.reshape(-1)
            if len(pred) >= horizon:
                return pred[:horizon].astype(float)
            if len(pred):
                return np.concatenate([pred, np.full(horizon - len(pred), pred[-1])]).astype(float)
    except Exception as e:
        print(f"  TTM forward failed ({e}); fallback.")
    return forecast_fallback(y, horizon)


def forecast_ttm_mc_dropout(model, kind: str, y: np.ndarray, horizon: int,
                            context: int = 512, samples: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """MC-dropout forecast: (mean, std) over `samples` stochastic forwards.

    Turns on dropout at inference and runs the forward pass repeatedly —
    a cheap, model-agnostic uncertainty estimate (no distribution head
    required). std = None when the model can't be sampled (fallback).
    """
    if model is None or kind == "fallback":
        return forecast_fallback(y, horizon), None
    import torch

    y = np.asarray(y, dtype=np.float32)
    hist = y[-min(context, len(y)):]
    try:
        x = torch.tensor(hist).view(1, -1, 1)
        if next(model.parameters()).is_cuda:
            x = x.to(next(model.parameters()).device)
        # enable dropout stochasticity for the forward passes
        train_state = model.training
        model.eval()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
        preds = []
        with torch.no_grad():
            for _ in range(samples):
                out = model(past_values=x) if "past_values" in model.forward.__code__.co_varnames else model(x)
                if hasattr(out, "prediction_outputs"):
                    p = out.prediction_outputs
                elif isinstance(out, torch.Tensor):
                    p = out
                else:
                    p = out[0]
                p = p.detach().cpu().numpy().reshape(-1)
                preds.append(p[:horizon])
        model.train(train_state)
        P = np.stack(preds)
        mean = P.mean(0).astype(float)
        std = P.std(0).astype(float)
        return mean, std
    except Exception as e:
        print(f"  MC-dropout failed ({e}); point forecast only.")
        return forecast_ttm_univariate(model, kind, y, horizon, context=context), None


def rolling_iterative_forecast(model, kind: str, y: np.ndarray, horizon: int, step: int = 8, context: int = 512) -> np.ndarray:
    """Extend beyond native horizon by iterative append (rolling forecast)."""
    y = list(np.asarray(y, dtype=float))
    preds = []
    remaining = horizon
    while remaining > 0:
        h = min(step, remaining)
        chunk = forecast_ttm_univariate(model, kind, np.asarray(y), h, context=context)
        preds.extend(chunk.tolist())
        y.extend(chunk.tolist())
        remaining -= h
    return np.asarray(preds[:horizon], dtype=float)


def forecast_multivariate_close(
    model, kind: str, panel: pd.DataFrame, target: str, horizon: int, context: int = 512
) -> np.ndarray:
    """
    Multivariate tip: use peer closes as extra channels; forecast target channel.
    For fallback, just use target univariate. For TTM, pass multi-feature tensor when API allows.
    """
    if target not in panel.columns:
        raise ValueError(f"{target} not in panel")
    # Primary: target series
    y = panel[target].dropna().values.astype(float)
    if model is None or kind == "fallback":
        return forecast_fallback(y, horizon)
    # Try multi-feature input (batch, time, features)
    try:
        import torch

        cols = [c for c in panel.columns if panel[c].notna().sum() > len(panel) * 0.8]
        mat = panel[cols].ffill().bfill().values.astype(np.float32)
        mat = mat[-min(context, len(mat)):]
        # scale features
        mu, sd = mat.mean(axis=0), mat.std(axis=0)
        sd[sd == 0] = 1.0
        mat_z = (mat - mu) / sd
        x = torch.tensor(mat_z).unsqueeze(0)  # 1, T, F
        with torch.no_grad():
            out = model(past_values=x) if hasattr(model, "forward") else model(x)
            pred = getattr(out, "prediction_outputs", out)
            if not isinstance(pred, torch.Tensor):
                pred = pred[0]
            pred = pred.detach().cpu().numpy()
            # take target channel if present
            tidx = cols.index(target) if target in cols else 0
            if pred.ndim == 3:
                series = pred[0, :, tidx]
            elif pred.ndim == 2:
                series = pred[0] if pred.shape[0] == 1 else pred[:, tidx] if pred.shape[1] > tidx else pred.reshape(-1)
            else:
                series = pred.reshape(-1)
            series = series[:horizon]
            # inverse scale target
            series = series * sd[tidx] + mu[tidx]
            if len(series) < horizon:
                series = np.concatenate([series, np.full(horizon - len(series), series[-1])])
            return series.astype(float)
    except Exception as e:
        print(f"  multivariate path failed ({e}); univariate fallback")
        return forecast_ttm_univariate(model, kind, y, horizon, context=context)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status(args):
    print("Granite TTM forecast module")
    print(f"  prices : {PRICES_FILE.exists()}  stocks: {STOCKS_FILE.exists()}")
    model, kind = load_granite_model(args.model)
    print(f"  model  : {args.model}  backend={kind}")
    if PRICES_FILE.exists():
        p = load_ohlcv()
        print(f"  rows   : {len(p):,}  tickers={p['ticker'].nunique()}  "
              f"{p['date'].min().date()} → {p['date'].max().date()}")
    print(f"  portfolio: {portfolio_tickers()}")
    print(f"  first trades: { {k: v.date().isoformat() for k,v in first_trade_dates().items()} }")


def cmd_forecast(args):
    ticker_map = resolve_ticker_index_map(args)
    tickers = list(ticker_map.keys())
    indexes = sorted({ix for labs in ticker_map.values() for ix in labs})
    horizon = args.horizon
    context = args.context
    model, kind = load_granite_model(args.model)
    first_map = first_trade_dates()
    days_ago = getattr(args, "days_ago", None)
    # --days-ago takes priority over --from-first-trade; portfolio still defaults to first-trade when neither set
    use_first = bool(getattr(args, "from_first_trade", False))
    index_args = parse_index_list(getattr(args, "index", None))
    if days_ago is None and not use_first and index_args == ["portfolio"]:
        use_first = True
    if len(indexes) > 1:
        print(f"Multi-index run: {', '.join(indexes)}  ({len(tickers)} unique tickers)")

    if days_ago is not None:
        print(f"History window: last {int(days_ago)} observations (--days-ago, max 512)")
    elif use_first:
        print("First-trade anchors:")
        for t in tickers:
            d = first_map.get(t)
            print(f"  {t}: {d.date() if d is not None else 'n/a'}")

    rows = []
    multivariate = getattr(args, "multivariate", False)
    use_rolling = getattr(args, "rolling", False)
    channels_mode = getattr(args, "channels", "close")

    peer_panel = None
    if multivariate and len(tickers) > 1:
        peer_panel = build_multivariate_bundle(tickers, mode="close_only")
        print(f"Multivariate peer panel: {peer_panel.shape}")

    # Regime-selected serving: plan which tickers have a pass6 checkpoint for
    # the current regime. Those get their regime model swapped in; the rest
    # keep the general model (honest degradation — regime selection is an
    # upgrade when available, never a downgrade).
    regime_now = current_regime()
    if getattr(args, "no_regime", False):
        serving_plan = {t: (None, None, "disabled") for t in tickers}
    else:
        serving_plan = {t: serve_regime_model(t) for t in tickers}
    n_served = sum(1 for (_, _, reason) in serving_plan.values() if reason == "served")
    n_cov = sum(1 for (_, _, reason) in serving_plan.values() if reason == "no_checkpoint")
    if regime_now:
        print(f"Regime serving: regime={regime_now}  served={n_served}  "
              f"checkpoint-missing={n_cov}  no-coverage={len(tickers)-n_served-n_cov}")
        stale = [(t, c.get("age_days")) for t, (p, c, r) in serving_plan.items()
                 if r == "served" and c and (c.get("age_days") or 0) > 90]
        if stale:
            print(f"  WARNING: stale regime checkpoints (>90d): "
                  f"{', '.join(f'{t} ({d}d)' for t, d in stale)} — retrain with pass6 --channels")

    for t in tickers:
        # Regime-selected serving + ensemble: when a regime checkpoint exists,
        # forecast with BOTH the general model and the regime model, then
        # average (equal-weight ensemble). Without a checkpoint, use the
        # general model alone.
        ckpt_path, reg_cfg, reg_reason = serving_plan.get(t, (None, None, "no_coverage"))
        regime_model = None
        if ckpt_path is not None and model is not None and kind != "fallback":
            try:
                import copy as _copy
                import torch as _torch
                from transformers import AutoConfig
                state = _torch.load(ckpt_path, map_location="cpu")
                ckpt_ch = int(state.get("n_channels", 1))
                ckpt_rpt = bool(state.get("rpt", False))
                ckpt_exog = bool(state.get("exog", False)) or ckpt_ch >= 4
                base_cfg = AutoConfig.from_pretrained(gd.DEFAULT_MODEL)
                base_cfg.num_input_channels = ckpt_ch
                if ckpt_rpt:
                    base_cfg.resolution_prefix_tuning = True
                    base_cfg.frequency_token_vocab_size = 5
                if ckpt_ch > 1 or ckpt_rpt:
                    # rebuild with the checkpoint's exact architecture (channel
                    # count / RPT) so the state dict matches — deepcopying the
                    # general 1-channel model would size-mismatch silently.
                    regime_model = type(model).from_pretrained(
                        gd.DEFAULT_MODEL, config=base_cfg, ignore_mismatched_sizes=True)
                else:
                    regime_model = _copy.deepcopy(model)
                regime_model.load_state_dict(state["model"], strict=False)
                regime_model.eval()
                regime_model = regime_model.to(next(model.parameters()).device)
                # stash the checkpoint's flags so the inference path can pass
                # the daily freq token (RPT) and build exog channels
                regime_model._rpt = ckpt_rpt
                regime_model._exog = ckpt_exog
            except Exception as e:
                print(f"  {t}: regime model load failed ({e}); general model")
                regime_model = None
        # Resolve start: days-ago > first-trade > full history
        start = None
        try:
            # Prefer resolving against the ticker's full date index when available
            prices_all = load_ohlcv_with_sectors([t])
            idx_all = (
                prices_all[prices_all["ticker"] == t]
                .assign(date=lambda d: pd.to_datetime(d["date"]))
                .sort_values("date")["date"]
            )
            start = resolve_history_start(
                t,
                pd.DatetimeIndex(idx_all) if len(idx_all) else None,
                days_ago=days_ago,
                from_first_trade=use_first and days_ago is None,
                first_map=first_map,
            )
        except Exception:
            start = resolve_history_start(
                t,
                None,
                days_ago=days_ago,
                from_first_trade=use_first and days_ago is None,
                first_map=first_map,
            )
        # Build feature panel for this ticker
        if channels_mode == "full":
            panel = build_panel(t, start=start)
            if panel.empty or len(panel) < 10:
                panel = build_panel(t, start=None)
            if panel.empty:
                print(f"  {t}: no panel")
                continue
            y = panel["close"].dropna().values.astype(float)
            hist_index = panel["close"].dropna().index
        else:
            prices = load_ohlcv_with_sectors([t])
            sub = prices[prices["ticker"] == t].set_index("date")["close"].sort_index().dropna()
            if start is not None:
                # index holds datetime.date values; normalize start to date
                start_d = pd.Timestamp(start).date()
                mask = [(d.date() if hasattr(d, "date") else d) >= start_d for d in sub.index]
                sub2 = sub[mask]
                if len(sub2) >= 10:
                    sub = sub2
                else:
                    reason = "days-ago" if days_ago is not None else "post-trade"
                    print(f"  {t}: {reason} short ({len(sub2)}); using full history n={len(sub)}")
            if len(sub) < 10:
                print(f"  {t}: skip n={len(sub)}")
                continue
            y = sub.values.astype(float)
            hist_index = sub.index
            panel = sub.to_frame("close")

        if args.log:
            y_in = np.log(np.clip(y, 1e-6, None))
        else:
            y_in = y

        use_exog = getattr(args, "exog", False)
        if use_exog:
            try:
                exog = build_exog_panel()
                # attach target close as primary channel + exog
                base = pd.DataFrame({"close": pd.Series(y, index=hist_index)})
                enriched = merge_exog(base, exog)
                if multivariate and peer_panel is not None:
                    # peers + exog
                    enriched = peer_panel.join(exog, how="left")
                    for c in exog.columns:
                        enriched[c] = enriched[c].ffill()
                pred = forecast_multivariate_close(
                    model, kind, enriched.dropna(how="all"), "close" if "close" in enriched.columns else t,
                    horizon, context=context,
                )
            except Exception as e:
                print(f"  exog path failed ({e}); univariate")
                pred = forecast_ttm_univariate(model, kind, y_in, horizon, context=context)
        elif multivariate and peer_panel is not None and t in peer_panel.columns:
            pred = forecast_multivariate_close(model, kind, peer_panel, t, horizon, context=context)
        elif use_rolling:
            pred = rolling_iterative_forecast(model, kind, y_in, horizon, step=min(8, horizon), context=context)
        else:
            use_mc = getattr(args, "uncertainty", False)
            if use_mc and kind != "fallback":
                pred, pred_std = forecast_ttm_mc_dropout(model, kind, y_in, horizon, context=context)
            else:
                pred = forecast_ttm_univariate(model, kind, y_in, horizon, context=context)
                pred_std = None
            # ensemble: average with the regime-selected model when available
            if regime_model is not None and kind != "fallback":
                n_ch = int(reg_cfg.get("n_channels", 1)) if reg_cfg else 1
                ckpt_exog = bool(getattr(regime_model, "_exog", False)) or n_ch >= 4
                if n_ch > 1:
                    # build (close, return, vol20[, exog]) channels for the regime model
                    y3 = np.stack(_channels_from_series(y_in), axis=-1) if y_in.ndim == 1 else y_in
                    if ckpt_exog and hist_index is not None and len(hist_index):
                        ex = _event_proximity_series(len(y_in), pd.Timestamp(hist_index[-1]))
                        y3 = np.concatenate([y3, ex[:, None]], axis=-1)
                    pred_reg = forecast_ttm_univariate(regime_model, "regime", y3, horizon, context=context)
                else:
                    pred_reg = forecast_ttm_univariate(regime_model, "regime", y_in, horizon, context=context)
                pred = 0.5 * pred + 0.5 * pred_reg
                kind = "ensemble"

        if args.log:
            pred = np.exp(pred)
            if "pred_std" in dir() and pred_std is not None:
                pred_std = pred_std * pred  # log-normal approx: std scales with exp(mean)

        last_date = hist_index[-1]
        last_price = float(y[-1])
        entry = float(y[0])
        since = (last_price / entry - 1) * 100
        future_idx = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)

        print(f"\n{t}  last={last_price:.2f} @ {last_date if hasattr(last_date, 'date') else last_date}  n={len(y)}  "
              f"since_start={since:+.1f}%  backend={kind}"
              f"{'  [multivariate]' if multivariate else ''}"
              f"{'  [rolling]' if use_rolling else ''}"
              f"{'  [exog]' if getattr(args, 'exog', False) else ''}")
        for h, (dt, p) in enumerate(zip(future_idx, pred), 1):
            chg = (p / last_price - 1) * 100
            print(f"  H+{h:02d} {dt.date()}  {p:8.2f}  ({chg:+.1f}%)")
            rows.append({
                "ticker": t,
                "index_name": index_label_for(t, ticker_map),
                "as_of": last_date,
                "first_trade_date": first_map.get(t, hist_index[0]),
                "history_start": hist_index[0],
                "history_n": len(y),
                "entry_close": round(entry, 4),
                "return_since_entry_pct": round(since, 3),
                "horizon": h,
                "forecast_date": dt,
                "forecast_close": round(float(p), 4),
                "forecast_std": round(float(pred_std[h - 1]), 4) if "pred_std" in dir() and pred_std is not None and h - 1 < len(pred_std) else None,
                "last_close": last_price,
                "pct_change": round(chg, 3),
                "model": args.model if kind != "fallback" else "statistical_fallback",
                "backend": kind,
                "regime_model": "regime" if (ckpt_path is not None) else ("none" if reg_reason == "no_coverage" else reg_reason),
                "regime_dir_acc": round(float(reg_cfg["dir_acc"]), 1) if reg_cfg and reg_cfg.get("dir_acc") is not None else None,
                "regime_excess": round(float(reg_cfg["dir_acc"] - reg_cfg["pers_dir"]), 1)
                                 if reg_cfg and reg_cfg.get("dir_acc") is not None and reg_cfg.get("pers_dir") is not None else None,
                "regime_ckpt_age_days": reg_cfg.get("age_days") if reg_cfg and reg_cfg.get("age_days") is not None else None,
                **({f"regime_dir_h{s}": reg_cfg.get(f"dir_acc_h{s}")
                    for s in (10, 21, 42, 63, 96)
                    if reg_cfg and reg_cfg.get(f"dir_acc_h{s}") is not None}),
                "multivariate": bool(multivariate),
                "rolling": bool(use_rolling),
                "channels": channels_mode,
            })

    if not rows:
        print("No forecasts produced.")
        return
    out = pd.DataFrame(rows)
    out.to_csv(FORECAST_CSV, index=False)
    try:
        out.to_parquet(FORECAST_FILE, index=False)
        print(f"\nWrote {FORECAST_FILE}")
    except Exception:
        pass
    print(f"Wrote {FORECAST_CSV} ({len(out)} rows)")


def directional_accuracy(actual: np.ndarray, pred: np.ndarray, last: float) -> float:
    """Fraction of steps where predicted direction matches realized direction."""
    if len(actual) == 0:
        return float("nan")
    a_dir = np.sign(actual - last)
    p_dir = np.sign(pred[: len(actual)] - last)
    # subsequent steps: vs previous actual
    if len(actual) > 1:
        a_dir = np.sign(np.diff(np.concatenate([[last], actual])))
        p_dir = np.sign(np.diff(np.concatenate([[last], pred[: len(actual)]])))
    return float((a_dir == p_dir).mean())


def cmd_backtest(args):
    """Rolling forecast backtest.

    --from-first-trade means *forecast origins* start at each ticker's first
    filled trade (holdings performance window). Pre-trade history may still
    feed model context so the test is: "from the day I held the asset, how
    good were forecasts?"
    """
    ticker_map = resolve_ticker_index_map(args)
    tickers = list(ticker_map.keys())
    indexes = sorted({ix for labs in ticker_map.values() for ix in labs})
    horizon = args.horizon
    window = args.window
    model, kind = load_granite_model(args.model)
    first_map = first_trade_dates()
    use_first = bool(getattr(args, "from_first_trade", False))
    if not use_first and indexes == ["portfolio"]:
        use_first = True
    metrics = []
    if len(indexes) > 1:
        print(f"Multi-index backtest: {', '.join(indexes)}  ({len(tickers)} unique tickers)")
    if use_first:
        print("Forecast origins restricted to on/after first trade dates (context may use earlier history)")

    for t in tickers:
        prices = load_ohlcv_with_sectors([t])
        s = prices[prices["ticker"] == t].set_index("date")["close"].sort_index().dropna()
        if len(s) < window + horizon + 5:
            print(f"  {t}: insufficient history ({len(s)})")
            continue
        y = s.values.astype(float)
        dates = s.index
        # earliest origin index such that hist window ends at origin+window-1
        # and we evaluate forecast starting at origin+window
        min_origin = 0
        if use_first and t in first_map:
            ft = first_map[t]
            # first date index on/after first trade where we still have `window` prior points
            eligible = np.where(dates >= ft)[0]
            if len(eligible) == 0:
                print(f"  {t}: no bars on/after first trade {ft.date()}")
                continue
            # origin is start of history window; forecast starts at origin+window
            # require forecast start date >= first trade
            # origin + window >= first_trade_idx  => origin >= first_trade_idx - window
            ft_idx = int(eligible[0])
            min_origin = max(0, ft_idx - window)
            print(f"  {t}: first_trade={ft.date()}  origin_from={dates[min(min_origin+window, len(dates)-1)].date()}")

        abs_err, sq_err, pct_err, dirs = [], [], [], []
        n_origins = 0
        step = max(1, horizon // 2)
        start_range = min_origin
        end_range = len(y) - horizon
        # need origin+window <= len-horizon effectively: origin <= len - window - horizon
        end_range = len(y) - window - horizon
        if end_range < start_range:
            print(f"  {t}: no room for window={window} horizon={horizon} after first-trade constraint")
            continue
        for origin in range(start_range, end_range + 1, step):
            hist = y[origin: origin + window]
            actual = y[origin + window: origin + window + horizon]
            if len(hist) < window or len(actual) < horizon:
                continue
            # enforce forecast start on/after first trade when requested
            if use_first and t in first_map:
                fc_start = dates[origin + window]
                if fc_start < first_map[t]:
                    continue
            pred = forecast_ttm_univariate(model, kind, hist, horizon, context=args.context)
            err = pred - actual
            abs_err.extend(np.abs(err))
            sq_err.extend(err ** 2)
            pct_err.extend(np.abs(err) / np.clip(np.abs(actual), 1e-6, None) * 100)
            dirs.append(directional_accuracy(actual, pred, hist[-1]))
            n_origins += 1

        if not abs_err:
            print(f"  {t}: no scored origins")
            continue
        metrics.append({
            "ticker": t,
            "index_name": index_label_for(t, ticker_map),
            "first_trade_date": first_map.get(t, pd.NaT),
            "from_first_trade": use_first,
            "horizon": horizon,
            "window": window,
            "n_origins": n_origins,
            "mae": round(float(np.mean(abs_err)), 4),
            "rmse": round(float(np.sqrt(np.mean(sq_err))), 4),
            "mape_pct": round(float(np.mean(pct_err)), 3),
            "directional_accuracy": round(float(np.nanmean(dirs)), 3),
            "backend": kind,
            "model": args.model if kind != "fallback" else "statistical_fallback",
        })
        m = metrics[-1]
        print(f"{t}: MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} MAPE={m['mape_pct']:.2f}% "
              f"DirAcc={m['directional_accuracy']:.1%} origins={m['n_origins']} "
              f"from_first_trade={use_first} ({kind})")

    if metrics:
        pd.DataFrame(metrics).to_csv(BACKTEST_FILE, index=False)
        print(f"Wrote {BACKTEST_FILE}")



def main():
    parser = argparse.ArgumentParser(
        description="Granite TTM stock forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("forecast")
    add_ticker_args(p)
    add_index_args(p)
    add_sector_arg(p)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--log", action="store_true")
    p.add_argument("--from-first-trade", action="store_true",
                   help="Start forecasting window at first filled trade (history context may include pre-trade bars; default for --index portfolio). Use with backtest to score reliability while held.")
    p.add_argument("--days-ago", type=int, default=None, metavar="N",
                   help="Use only the last N history points (1..512). Overrides --from-first-trade.")
    p.add_argument("--multivariate", action="store_true",
                   help="Use peer closes as extra channels")
    p.add_argument("--rolling", action="store_true",
                   help="Iterative rolling forecast for long horizons")
    p.add_argument("--channels", choices=["close", "full"], default="close",
                   help="close=price only; full=OHLCV+indicators panel")
    p.add_argument("--exog", action="store_true",
                   help="Merge exogenous channels (mkt return, vol, sector, dispersion)")
    p.add_argument("--no-regime", action="store_true",
                   help="Disable regime-selected model serving (default: ON when checkpoints exist)")
    p.add_argument("--uncertainty", action="store_true",
                   help="Emit MC-dropout std band (forecast_std column) on forecasts")
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("backtest")
    add_ticker_args(p)
    add_index_args(p)
    add_sector_arg(p)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--from-first-trade", action="store_true",
                   help="Only score forecast origins on/after first filled trade; "
                        "pre-trade bars may still provide model context")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("list-models")
    p.set_defaults(func=lambda a: print(
        "ibm-granite/granite-timeseries-ttm-r2 (default)\n"
        "ibm-granite/granite-timeseries-ttm-r1\n"
        "ibm-granite/granite-timeseries-ttm-v1"
    ))

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
