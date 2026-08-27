#!/usr/bin/env python3
"""granite_service.py — Live Granite/fallback forecast microservice for the dashboard.

Always computes forecasts on request (never serves stale forecast parquet as the answer).
Supports multivariate channels, index peers, correlated/uncorrelated peer sets,
days-ago history windows, and multi-index membership via index_registry.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))

from index_registry import available_indexes, parse_indexes, tickers_for_index  # noqa: E402

HOST, PORT = "0.0.0.0", 5055


def _json(handler, code, payload):
    body = json.dumps(payload, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def _portfolio():
    return tickers_for_index("portfolio")


def _indexes(name: str) -> list[str]:
    try:
        names = parse_indexes(name)
    except ValueError:
        names = [name]
    out, seen = [], set()
    for n in names:
        for t in tickers_for_index(n):
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _load_prices(tickers: list[str]):
    import pandas as pd
    try:
        from forecast_granite import load_ohlcv_with_sectors
        prices = load_ohlcv_with_sectors(tickers)
        if prices is not None and len(prices):
            prices["date"] = pd.to_datetime(prices["date"])
            return prices
    except Exception:
        pass
    try:
        from ttm_features import load_ohlcv
        prices = load_ohlcv(tickers)
        if prices is not None and len(prices):
            prices["date"] = pd.to_datetime(prices["date"])
            return prices
    except Exception:
        pass
    import pandas as pd
    prices = pd.read_parquet(DATA_DIR / "daily_prices/")
    prices["date"] = pd.to_datetime(prices["date"])
    if tickers:
        prices = prices[prices["ticker"].isin(tickers)]
    sp = DATA_DIR / "sector_prices.parquet"
    sect = [t for t in tickers if str(t).startswith("SECT_")]
    if sect and sp.exists():
        sdf = pd.read_parquet(sp)
        sdf["date"] = pd.to_datetime(sdf["date"])
        prices = pd.concat([prices, sdf[sdf["ticker"].isin(sect)]], ignore_index=True)
    return prices


def _corr_peer_sets(target: str, candidate_tickers: list[str], prices, top_n: int = 5):
    """Return (correlated, uncorrelated) peer lists vs target from candidate universe."""
    import numpy as np
    import pandas as pd
    wide = (
        prices[prices["ticker"].isin([target] + list(candidate_tickers))]
        .pivot_table(index="date", columns="ticker", values="close")
        .sort_index()
        .ffill()
    )
    if target not in wide.columns or len(wide) < 30:
        return [], []
    rets = np.log(wide / wide.shift(1)).dropna(how="all")
    if target not in rets.columns:
        return [], []
    corr = rets.corr()[target].drop(labels=[target], errors="ignore").dropna()
    if corr.empty:
        return [], []
    corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)
    correlated = corr_sorted.head(top_n).index.tolist()
    # least absolute correlation
    uncorrelated = corr.reindex(corr.abs().sort_values(ascending=True).index).head(top_n).index.tolist()
    return correlated, uncorrelated


def _live_forecast(
    tickers: list[str],
    horizon: int = 10,
    *,
    from_first_trade: bool = False,
    days_ago: int | None = None,
    multivariate: bool = False,
    peer_mode: str = "none",  # none | index | correlated | uncorrelated | custom
    peer_tickers: list[str] | None = None,
    peer_index: str | None = None,
    peer_top_n: int = 5,
    context: int = 512,
    use_log: bool = False,
    base: str = "ibm",  # "ibm" | "rpt"
    no_regime: bool = False,
):
    """Compute live forecasts — always model/fallback, never stored forecast files."""
    import numpy as np
    import pandas as pd
    from forecast_granite import (
        first_trade_dates,
        forecast_ttm_univariate,
        forecast_multivariate_close,
        load_granite_model,
        resolve_history_start,
    )

    t0 = time.time()
    horizon = max(1, min(int(horizon), 96))
    context = max(32, min(int(context), 512))
    tickers = [t.strip().upper() for t in tickers if t and str(t).strip()]
    if not tickers:
        return {"ok": False, "error": "no tickers"}

    model, kind = load_granite_model()
    first_map = first_trade_dates()

    # peer universe for multivariate
    peer_pool: list[str] = []
    if peer_mode == "custom" and peer_tickers:
        peer_pool = [t.strip().upper() for t in peer_tickers if t.strip()]
    elif peer_mode == "index" and peer_index:
        peer_pool = _indexes(peer_index)
    elif peer_mode in ("correlated", "uncorrelated"):
        # candidates: peer_index if set else all defensive+growth+portfolio
        if peer_index:
            peer_pool = _indexes(peer_index)
        else:
            peer_pool = list(dict.fromkeys(
                _indexes("portfolio") + _indexes("defensive") + _indexes("growth")
            ))

    all_needed = list(dict.fromkeys(tickers + peer_pool))
    prices = _load_prices(all_needed)

    rows = []
    charts = {}
    errors = []

    for t in tickers:
        try:
            sub = (
                prices[prices["ticker"] == t]
                .set_index("date")["close"]
                .sort_index()
                .dropna()
            )
            if len(sub) < 10:
                errors.append({"ticker": t, "error": f"insufficient history n={len(sub)}"})
                continue

            start = resolve_history_start(
                t,
                pd.DatetimeIndex(sub.index),
                days_ago=days_ago,
                from_first_trade=from_first_trade and days_ago is None,
                first_map=first_map,
            )
            if start is not None:
                clipped = sub[sub.index >= start]
                if len(clipped) >= 10:
                    sub = clipped

            y = sub.values.astype(float)
            hist_index = sub.index
            y_in = np.log(np.clip(y, 1e-6, None)) if use_log else y

            peers_used: list[str] = []
            pred = None
            if multivariate and peer_mode != "none":
                if peer_mode == "correlated":
                    peers_used, _ = _corr_peer_sets(t, [p for p in peer_pool if p != t], prices, peer_top_n)
                elif peer_mode == "uncorrelated":
                    _, peers_used = _corr_peer_sets(t, [p for p in peer_pool if p != t], prices, peer_top_n)
                elif peer_mode in ("index", "custom"):
                    peers_used = [p for p in peer_pool if p != t][: max(peer_top_n * 2, peer_top_n)]

                if peers_used:
                    wide = (
                        prices[prices["ticker"].isin([t] + peers_used)]
                        .pivot_table(index="date", columns="ticker", values="close")
                        .sort_index()
                        .ffill()
                    )
                    if start is not None:
                        wide = wide[wide.index >= start]
                    wide = wide.dropna(how="all")
                    if t in wide.columns and len(wide) >= 10:
                        try:
                            pred = forecast_multivariate_close(
                                model, kind, wide, t, horizon, context=context
                            )
                        except Exception:
                            pred = None

            if pred is None:
                pred = forecast_ttm_univariate(model, kind, y_in, horizon, context=context)
                if use_log:
                    pred = np.exp(pred)

            last_date = hist_index[-1]
            last_price = float(y[-1])
            future_idx = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
            hist_pts = [
                {"date": d.strftime("%Y-%m-%d"), "close": float(v)}
                for d, v in zip(hist_index[-60:], y[-60:])
            ]
            fc_pts = []
            for h, (dt, pv) in enumerate(zip(future_idx, pred), 1):
                chg = (float(pv) / last_price - 1) * 100
                rec = {
                    "ticker": t,
                    "horizon": h,
                    "forecast_date": dt.strftime("%Y-%m-%d"),
                    "forecast_close": round(float(pv), 4),
                    "last_close": round(last_price, 4),
                    "pct_change": round(chg, 3),
                    "backend": kind,
                    "as_of": last_date.strftime("%Y-%m-%d"),
                    "history_n": len(y),
                    "multivariate": bool(multivariate and peers_used),
                    "peers": peers_used,
                }
                rows.append(rec)
                fc_pts.append({
                    "date": rec["forecast_date"],
                    "close": rec["forecast_close"],
                    "horizon": h,
                    "pct_change": rec["pct_change"],
                })

            charts[t] = {
                "history": hist_pts,
                "forecast": fc_pts,
                "last_close": round(last_price, 4),
                "as_of": last_date.strftime("%Y-%m-%d"),
                "backend": kind,
                "peers": peers_used,
                "history_n": len(y),
            }
        except Exception as e:
            errors.append({"ticker": t, "error": str(e)})

    return {
        "ok": True,
        "live": True,
        "horizon": horizon,
        "tickers": tickers,
        "from_first_trade": bool(from_first_trade),
        "days_ago": days_ago,
        "multivariate": bool(multivariate),
        "peer_mode": peer_mode,
        "peer_index": peer_index,
        "context": context,
        "backend": kind,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "forecasts": rows,
        "charts": charts,
        "errors": errors,
        "available_indexes": available_indexes(),
    }


def handle_forecast(qs, body=None):
    body = body or {}
    def g(key, default=None):
        if key in body and body[key] is not None:
            return body[key]
        v = qs.get(key, [default])
        return v[0] if isinstance(v, list) else v

    tickers_raw = g("tickers") or g("ticker")
    if tickers_raw in (None, "null", ""):
        tickers_raw = None
    horizon = int(g("horizon", 10) or 10)
    from_first = str(g("from_first_trade", "0")).lower() in ("1", "true", "yes")
    days_ago = g("days_ago")
    days_ago = int(days_ago) if days_ago not in (None, "", "null") else None
    multivariate = str(g("multivariate", "0")).lower() in ("1", "true", "yes")
    peer_mode = str(g("peer_mode", "none") or "none").lower()
    peer_index = g("peer_index") or g("index")
    peer_top_n = int(g("peer_top_n", 5) or 5)
    context = int(g("context", 512) or 512)
    use_log = str(g("log", "0")).lower() in ("1", "true", "yes")
    base = g("base", "ibm")
    no_regime = str(g("no_regime", "0")).lower() in ("1", "true", "yes")
    peer_tickers_raw = g("peer_tickers") or ""

    index_name = g("index_name") or g("name") or g("index")
    if not tickers_raw and index_name:
        tickers = _indexes(str(index_name))
    elif tickers_raw:
        if isinstance(tickers_raw, list):
            tickers = [str(x).strip().upper() for x in tickers_raw]
        else:
            tickers = [x.strip().upper() for x in str(tickers_raw).split(",") if x.strip()]
    else:
        return {"ok": False, "error": "tickers or index required"}

    peer_tickers = []
    if peer_tickers_raw:
        if isinstance(peer_tickers_raw, list):
            peer_tickers = [str(x).strip().upper() for x in peer_tickers_raw]
        else:
            peer_tickers = [x.strip().upper() for x in str(peer_tickers_raw).split(",") if x.strip()]

    try:
        return _live_forecast(
            tickers,
            horizon,
            from_first_trade=from_first,
            days_ago=days_ago,
            multivariate=multivariate,
            peer_mode=peer_mode,
            peer_tickers=peer_tickers,
            peer_index=str(peer_index) if peer_index else None,
            peer_top_n=peer_top_n,
            context=context,
            use_log=use_log,
            base=base,
            no_regime=no_regime,
        )
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)
            if path == "/health":
                return _json(self, 200, {
                    "ok": True,
                    "service": "granite-forecast",
                    "live": True,
                    "data_dir": str(DATA_DIR),
                    "indexes": available_indexes(),
                })
            if path == "/indexes":
                return _json(self, 200, {
                    "ok": True,
                    "indexes": available_indexes(),
                    "members": {n: tickers_for_index(n)[:20] for n in available_indexes()},
                })
            if path == "/tickers":
                try:
                    import pandas as pd
                    tickers = pd.read_parquet(DATA_DIR / "monitored_stocks.parquet")["ticker"].tolist()
                except Exception:
                    tickers = []
                return _json(self, 200, {"ok": True, "tickers": tickers})
            if path == "/forecast/portfolio":
                qs = {**qs, "tickers": [",".join(_portfolio())], "from_first_trade": qs.get("from_first_trade", ["1"])}
                return _json(self, 200, handle_forecast(qs))
            if path == "/forecast/index":
                name = qs.get("name", qs.get("index", ["fertilizer"]))[0]
                qs = {**qs, "tickers": [",".join(_indexes(name))], "index_name": [name]}
                return _json(self, 200, handle_forecast(qs))
            if path == "/forecast/sectors":
                qs = {**qs, "tickers": [",".join(tickers_for_index("sectors"))]}
                return _json(self, 200, handle_forecast(qs))
            if path == "/forecast":
                return _json(self, 200, handle_forecast(qs))
            return _json(self, 404, {"ok": False, "error": f"unknown path {path}"})
        except Exception as e:
            return _json(self, 500, {"ok": False, "error": str(e), "trace": traceback.format_exc()})

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode() or "{}")
            except Exception:
                body = {}
            if path in ("/forecast", "/forecast/live"):
                return _json(self, 200, handle_forecast(qs, body))
            return _json(self, 404, {"ok": False, "error": f"unknown path {path}"})
        except Exception as e:
            return _json(self, 500, {"ok": False, "error": str(e), "trace": traceback.format_exc()})


def main():
    ap = argparse.ArgumentParser(description="Live Granite forecast microservice")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    print(f"granite_service (LIVE) on http://{args.host}:{args.port}", flush=True)
    print(f"  indexes: {available_indexes()}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
