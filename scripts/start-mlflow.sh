#!/bin/sh
set -eu

MLFLOW_DATA_DIR="${MLFLOW_DATA_DIR:-/tmp/mlflow}"
MLFLOW_ARTIFACT_DIR="${MLFLOW_ARTIFACT_DIR:-$MLFLOW_DATA_DIR/artifacts}"
mkdir -p "$MLFLOW_ARTIFACT_DIR"

exec mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI:-sqlite:///$MLFLOW_DATA_DIR/mlflow.db}" \
  --artifacts-destination "$MLFLOW_ARTIFACT_DIR" \
  --serve-artifacts \
  --host 0.0.0.0 \
  --workers "${MLFLOW_WORKERS:-1}" \
  --port "${PORT:-5000}"
