#!/usr/bin/env python3
"""
pipeline_service.py — HTTP control plane to rerun data jobs for the dashboard.

POST /run  {"job": "prices"|"backfill"|"analytics"|"export"|"forecast_bt"|"monte_carlo"|"all"|...}
GET  /jobs
GET  /status
GET  /health

Jobs are subprocesses of existing programs; logs under logs/pipeline_*.log
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DATA_DIR = Path(__file__).resolve().parent
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
HOST, PORT = "127.0.0.1", 5056

# name -> argv (python script relative to DATA_DIR)
JOB_CATALOG = {
    "prices": ["update_prices.py"],
    "backfill": ["backfill_historical.py", "--period", "1y"],
    "analytics": ["run_daily_automation.py"],
    "export": ["export_dashboard_data.py"],
    "catalog": ["build_data_catalog.py"],
    "manage_list": ["manage_stocks.py", "list"],

    "data_integrity": ["data_integrity.py", "all", "--save"],
    "rebalance_cal": ["rebalance_calendar.py", "--months", "18", "--save"],
    "risk_ext": ["risk_metrics_ext.py", "--save"],
    "research_hygiene": ["research_hygiene.py", "all", "--save"],
    "bl_views": ["black_litterman_views.py", "--save"],
    "momentum": ["momentum_analytics.py", "--universe", "all", "--save"],
    "factor_panel": ["factor_panel.py", "--save"],
    "buy_candidates": ["buy_candidates.py", "--save"],
    "fisher_baskets": ["fisher_sector_baskets.py", "--index", "sp500", "--save"],
    "integrity_deep": ["data_integrity_deep.py", "--save"],
    "manage_apply": ["manage_stocks.py", "apply-json", "--file", "staged_stock_updates.json"],

    "preferred": ["preferred_metrics.py", "--save"],
    "inclusion": ["inclusion_criteria.py", "--save"],
    "stress": ["stress_dual_pass.py", "--save"],
    "hmm": ["hmm_regime_detection.py", "--save"],
    "monte_carlo": ["monte_carlo.py", "--index", "portfolio", "--n-paths", "2000", "--save"],
    "mcmc": ["mcmc_regimes.py", "--index", "portfolio", "--save"],
    "forecast_bt": ["forecast_granite.py", "backtest", "--index", "portfolio", "--from-first-trade", "--horizon", "10", "--window", "60"],
    "forecast_reliability": ["forecast_reliability.py", "--index", "portfolio", "--save"],
    "fisher": ["fisher_index.py", "--index", "portfolio", "--save"],
    "index_bt": ["live_index_backtest.py", "--years", "1", "--json"],
    "fisher_backfill": ["fisher_index.py", "--backfill-all", "--years", "2", "--save"],
    "forecast_bt_param": ["forecast_granite.py", "backtest", "--index", "portfolio", "--from-first-trade"],
}

_state = {
    "running": None,  # job name
    "last": None,
    "history": [],  # recent job results
}
_lock = threading.Lock()


def _run_job(name: str, extra_args: list[str] | None = None) -> dict:
    if name == "all":
        results = []
        for key in ["prices", "analytics", "export"]:
            results.append(_run_job(key))
        return {"ok": all(r.get("ok") for r in results), "job": "all", "steps": results}

    if name not in JOB_CATALOG:
        return {"ok": False, "error": f"unknown job {name}", "available": list(JOB_CATALOG)}

    argv = [sys.executable, str(DATA_DIR / JOB_CATALOG[name][0])] + JOB_CATALOG[name][1:]
    if extra_args:
        argv.extend(extra_args)
    log_path = LOG_DIR / f"pipeline_{name}_{int(time.time())}.log"
    t0 = time.time()
    with _lock:
        if _state["running"]:
            return {"ok": False, "error": f"busy: {_state['running']} running"}
        _state["running"] = name
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(
                argv,
                cwd=str(DATA_DIR),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        log_tail = ""
        payload = None
        try:
            text = log_path.read_text(errors="replace")
            log_tail = text[-2000:]
            # Prefer last JSON object if script used --json
            for line in reversed(text.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        payload = json.loads(line)
                        break
                    except Exception:
                        pass
        except Exception:
            pass
        result = {
            "ok": proc.returncode == 0,
            "job": name,
            "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - t0, 2),
            "log": str(log_path),
            "log_tail": log_tail[-500:] if log_tail else "",
            "cmd": argv,
            "result": payload,
        }
    except subprocess.TimeoutExpired:
        result = {"ok": False, "job": name, "error": "timeout", "log": str(log_path)}
    except Exception as e:
        result = {"ok": False, "job": name, "error": str(e)}
    with _lock:
        _state["running"] = None
        _state["last"] = result
        _state["history"] = ([result] + _state["history"])[:20]
    return result


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
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/health":
            return _json(self, 200, {"ok": True, "service": "pipeline", "running": _state["running"]})
        if path == "/jobs":
            return _json(self, 200, {"ok": True, "jobs": {k: v for k, v in JOB_CATALOG.items()}})
        if path == "/status":
            return _json(self, 200, {"ok": True, **_state})
        return _json(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode() or "{}")
        except Exception:
            body = {}
        if path == "/run":
            job = body.get("job") or parse_qs(urlparse(self.path).query).get("job", [None])[0]
            extra = body.get("args") or []
            if not job:
                return _json(self, 400, {"ok": False, "error": "job required", "jobs": list(JOB_CATALOG)})

            # async option
            if body.get("async"):
                def bg():
                    _run_job(job, extra)
                threading.Thread(target=bg, daemon=True).start()
                return _json(self, 202, {"ok": True, "accepted": job, "async": True})

            result = _run_job(job, extra)
            return _json(self, 200 if result.get("ok") else 500, result)
        return _json(self, 404, {"ok": False, "error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    print(f"pipeline_service on http://{args.host}:{args.port}", flush=True)
    print(f"  jobs: {list(JOB_CATALOG)}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
