#!/usr/bin/env bash
# start_dashboard.sh — granite_service + pipeline_service + static dashboard
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PORT_API="${PORT_API:-5055}"
PORT_PIPE="${PORT_PIPE:-5056}"
PORT_WEB="${PORT_WEB:-8765}"
HOST="${HOST:-0.0.0.0}"
mkdir -p "$ROOT/logs" "$ROOT/dashboard_data"

echo "==> Export dashboard data (best effort)"
python3 export_dashboard_data.py 2>"$ROOT/logs/export_dashboard.err" || true
python3 build_data_catalog.py 2>"$ROOT/logs/data_catalog.err" || true

echo "==> granite_service (live forecasts) :${PORT_API}"
python3 granite_service.py --host "$HOST" --port "$PORT_API" >"$ROOT/logs/granite_service.log" 2>&1 &
echo $! > "$ROOT/logs/granite_service.pid"

echo "==> pipeline_service (job runner)    :${PORT_PIPE}"
python3 pipeline_service.py --host "$HOST" --port "$PORT_PIPE" >"$ROOT/logs/pipeline_service.log" 2>&1 &
echo $! > "$ROOT/logs/pipeline_service.pid"

PORT_ANALYTICS="${PORT_ANALYTICS:-8767}"
echo "==> analytics_service                  :${PORT_ANALYTICS}"
python3 analytics_service.py --port "$PORT_ANALYTICS" >"$ROOT/logs/analytics_service.log" 2>&1 &
echo $! > "$ROOT/logs/analytics_service.pid"

echo "==> static dashboard                 :${PORT_WEB}"
python3 -m http.server "$PORT_WEB" --bind "$HOST" >"$ROOT/logs/dashboard_http.log" 2>&1 &
echo $! > "$ROOT/logs/dashboard_http.pid"

cleanup() {
  echo "Shutting down..."
  kill $(cat "$ROOT/logs/granite_service.pid" 2>/dev/null) 2>/dev/null || true
  kill $(cat "$ROOT/logs/pipeline_service.pid" 2>/dev/null) 2>/dev/null || true
  kill $(cat "$ROOT/logs/dashboard_http.pid" 2>/dev/null) 2>/dev/null || true
  kill $(cat "$ROOT/logs/analytics_service.pid" 2>/dev/null) 2>/dev/null || true
}
trap cleanup EXIT INT TERM
sleep 1
echo ""
echo "Dashboard:   http://${HOST}:${PORT_WEB}/index.html"
echo "Forecasts:   http://${HOST}:${PORT_API}/health"
echo "Pipelines:   http://${HOST}:${PORT_PIPE}/jobs"
echo "Analytics:   http://${HOST}:${PORT_ANALYTICS:-8767}/health"
echo "Ctrl+C to stop."
wait
