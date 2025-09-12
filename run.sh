#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Activate local venv if present
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate || true
fi

if [ -f .env ]; then export $(grep -v '^#' .env | xargs); fi

# Runtime opts
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
DEV_RELOAD=${DEV_RELOAD:-0}

OPTS=(app.main:app --host "$HOST" --port "$PORT" --http h11 --loop asyncio --access-log)
if [ "$DEV_RELOAD" = "1" ]; then
  OPTS+=(--reload)
fi

# Kill anything bound to the target port (best-effort)
if command -v lsof >/dev/null 2>&1; then
  PIDS=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "[run.sh] Port $PORT busy. Killing PIDs: $PIDS" >&2
    # Try graceful TERM first
    kill $PIDS 2>/dev/null || true
    sleep 0.5
    # If still listening, force kill
    STILL=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -n "$STILL" ]; then
      echo "[run.sh] Forcing kill on: $STILL" >&2
      kill -9 $STILL 2>/dev/null || true
      sleep 0.2
    fi
  fi
fi

# Prefer venv uvicorn if available
if [ -x .venv/bin/uvicorn ]; then
  exec .venv/bin/uvicorn "${OPTS[@]}"
else
  exec uvicorn "${OPTS[@]}"
fi
