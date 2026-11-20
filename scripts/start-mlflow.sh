#!/bin/sh
set -eu

MLFLOW_DATA_DIR="${MLFLOW_DATA_DIR:-/tmp/mlflow}"
MLFLOW_ARTIFACT_DIR="${MLFLOW_ARTIFACT_DIR:-$MLFLOW_DATA_DIR/artifacts}"
mkdir -p "$MLFLOW_ARTIFACT_DIR"

mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI:-sqlite:///$MLFLOW_DATA_DIR/mlflow.db}" \
  --artifacts-destination "$MLFLOW_ARTIFACT_DIR" \
  --serve-artifacts \
  --allowed-hosts "*.onrender.com,localhost:*,127.0.0.1:*,asr-adperf-mlflow:5000" \
  --host 0.0.0.0 \
  --port "${PORT:-5000}" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' INT TERM

python - <<'PY'
import time
import urllib.error
import urllib.request

url = f"http://127.0.0.1:{__import__('os').environ.get('PORT', '5000')}/health"
for _ in range(60):
    try:
        urllib.request.urlopen(url, timeout=2).read()
        break
    except (OSError, urllib.error.URLError):
        time.sleep(2)
else:
    raise SystemExit("MLflow did not become healthy")
PY

python scripts/seed-mlflow.py || echo "MLflow lineage seed failed; tracker remains available."
wait "$server_pid"
