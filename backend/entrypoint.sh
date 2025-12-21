#!/usr/bin/env sh
set -eu

worker_pid=""

if [ "${RUN_WORKER:-}" = "1" ] || [ "${RUN_WORKER:-}" = "true" ]; then
  echo "[backend] starting embedded worker"
  python -m backend.worker.main &
  worker_pid="$!"
fi

shutdown() {
  if [ -n "${worker_pid}" ]; then
    kill "${worker_pid}" 2>/dev/null || true
  fi
}

trap shutdown INT TERM

exec uvicorn backend.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"

