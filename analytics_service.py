#!/usr/bin/env python3
"""
analytics_service.py — HTTP microservice for dashboard ops.

Endpoints:
  GET  /health
  GET  /tables                      — list available parquet/csv artifacts
  GET  /table?name=preferred_metrics
  GET  /dual-pass
  GET  /rolling?universe=portfolio&window=63
  GET  /aerospace
  POST /run/update-prices
  POST /run/backfill
  POST /run/fundamentals-snapshot
  POST /run/preferred-metrics
  POST /run/rolling
  POST /run/growth-analytics
  POST /run/alerts
  POST /run/export-dashboard
  POST /run/all-daily               — price path + metrics + export

  python analytics_service.py --port 8765
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DATA_DIR = Path(__file__).parent
PORT = 8765


def read_table(name: str):
    import pandas as pd
    # try parquet then csv
    for ext in (".parquet", ".csv"):
        p = DATA_DIR / f"{name}{ext}"
        if p.exists():
            if ext == ".parquet":
                try:
                    return pd.read_parquet(p)
                except Exception:
                    import duckdb
                    return duckdb.query(f"SELECT * FROM read_parquet('{p}')").df()
            return pd.read_csv(p)
    # aliases
    aliases = {
        "preferred": "preferred_metrics",
        "trifecta": "preferred_screen_hits",
        "dual": "preferred_metrics",
        "rolling": "rolling_window_metrics",
        "screen_stability": "rolling_screen_stability",
        "dupont": "dupont_analysis",
        "bl": "black_litterman_weights",
        "vol_rp": "vol_target_vs_risk_parity",
        "growth_vol": "growth_tech_vol_returns",
        "history": "preferred_metrics_history",
        "screen_bt": "screen_backtest",
        "fisher": "fisher_indexes",
    }
    if name in aliases:
        return read_table(aliases[name])
    return None


def df_json(df, limit=500):
    if df is None:
        return {"error": "not found"}
    if len(df) > limit:
        df = df.head(limit)
    return {"rows": len(df), "columns": list(df.columns), "data": json.loads(df.to_json(orient="records", date_format="iso"))}


def run_cmd(args: list[str], timeout: int = 120) -> dict:
    try:
        r = subprocess.run(
            [sys.executable, *args],
            cwd=str(DATA_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": r.returncode == 0,
            "returncode": r.returncode,
            "stdout": (r.stdout or "")[-4000:],
            "stderr": (r.stderr or "")[-2000:],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, payload):
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        try:
            if path == "/health":
                return self._json(200, {"status": "ok", "service": "analytics", "dir": str(DATA_DIR)})
            if path == "/tables":
                files = sorted(
                    [p.name for p in DATA_DIR.glob("*.csv")]
                    + [p.name for p in DATA_DIR.glob("*.parquet")]
                )
                return self._json(200, {"files": files})
            if path == "/table":
                name = qs.get("name", [""])[0].replace(".csv", "").replace(".parquet", "")
                limit = int(qs.get("limit", ["300"])[0])
                df = read_table(name)
                return self._json(200 if df is not None else 404, df_json(df, limit))
            if path == "/dual-pass":
                df = read_table("preferred_metrics")
                if df is None:
                    return self._json(404, {"error": "preferred_metrics missing — POST /run/preferred-metrics"})
                dual = df[df.get("decision") == "INCLUDE_CORE"] if "decision" in df.columns else df
                # also filter explicit dual
                if "buffett_pass" in df.columns and "trifecta_pass" in df.columns:
                    dual = df[(df.buffett_pass == True) & (df.trifecta_pass == True)]
                return self._json(200, df_json(dual, 100))
            if path == "/rolling":
                universe = qs.get("universe", ["portfolio"])[0]
                window = qs.get("window", ["63"])[0]
                run_cmd(["rolling_window_analysis.py", "--universe", universe, "--window", window, "--save"])
                return self._json(200, df_json(read_table("rolling_window_metrics")))
            if path == "/aerospace":
                stocks = read_table("monitored_stocks")
                if stocks is None:
                    return self._json(404, {"error": "no stocks"})
                mask = stocks.get("growth_sleeve", pd_series_empty()).isin(
                    ["aerospace_defense", "launch_services", "starlink_supply", "maritime_launch"]
                ) if "growth_sleeve" in stocks.columns else stocks.ticker.isna()
                sub = stocks[mask] if hasattr(mask, "any") else stocks
                return self._json(200, df_json(sub, 200))
            return self._json(404, {"error": f"unknown path {path}"})
        except Exception as e:
            return self._json(500, {"error": str(e), "trace": traceback.format_exc()[-1500:]})

    def do_POST(self):
        u = urlparse(self.path)
        path = u.path
        jobs = {
            "/run/update-prices": ["update_prices.py"],
            "/run/backfill": ["backfill_prices.py"] if (DATA_DIR / "backfill_prices.py").exists() else ["fundamentals_history.py", "backfill", "--quarters", "4"],
            "/run/fundamentals-snapshot": ["fundamentals_history.py", "snapshot"],
            "/run/preferred-metrics": ["preferred_metrics.py", "--save"],
            "/run/rolling": ["rolling_window_analysis.py", "--save"],
            "/run/growth-analytics": ["growth_tech_analytics.py"],
            "/run/dupont": ["dupont_analysis.py", "--save"],
            "/run/alerts": ["check_alerts.py", "--dry-run"],
            "/run/export-dashboard": ["export_dashboard_data.py"],
            "/run/data-integrity": ["data_integrity.py", "all", "--save"],
            "/run/rebalance-calendar": ["rebalance_calendar.py", "--save"],
            "/run/risk-ext": ["risk_metrics_ext.py", "--save"],
            "/run/research-hygiene": ["research_hygiene.py", "all", "--save"],
            "/run/bl-views": ["black_litterman_views.py", "--save"],
            "/run/screen-backtest": ["fundamentals_history.py", "backtest-screens"],
        }
        try:
            if path == "/run/all-daily":
                results = {}
                for key in ["/run/preferred-metrics", "/run/rolling", "/run/fundamentals-snapshot",
                            "/run/screen-backtest", "/run/export-dashboard"]:
                    args = jobs[key]
                    results[key] = run_cmd(args, timeout=180)
                return self._json(200, {"job": "all-daily", "results": results})
            if path in jobs:
                return self._json(200, {"job": path, **run_cmd(jobs[path], timeout=180)})
            return self._json(404, {"error": f"unknown job {path}", "known": list(jobs)})
        except Exception as e:
            return self._json(500, {"error": str(e), "trace": traceback.format_exc()[-1500:]})


def pd_series_empty():
    import pandas as pd
    return pd.Series(dtype=object)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    print(f"analytics_service on http://127.0.0.1:{args.port}")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
