#!/bin/sh
set -eu

MLFLOW_DATA_DIR="${MLFLOW_DATA_DIR:-/tmp/mlflow}"
MLFLOW_ARTIFACT_DIR="${MLFLOW_ARTIFACT_DIR:-$MLFLOW_DATA_DIR/artifacts}"
mkdir -p "$MLFLOW_ARTIFACT_DIR"

mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI:-sqlite:///$MLFLOW_DATA_DIR/mlflow.db}" \
  --artifacts-destination "$MLFLOW_ARTIFACT_DIR" \
  --serve-artifacts \
  --host 0.0.0.0 \
  --workers "${MLFLOW_WORKERS:-1}" \
  --port "${PORT:-5000}" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' INT TERM

# Seed through the running tracking API so the experiment uses the server's
# configured artifact proxy and the same active backend store.
export MLFLOW_TRACKING_URI="http://127.0.0.1:${PORT:-5000}"
python scripts/seed-mlflow-on-start.py

wait "$SERVER_PID"
