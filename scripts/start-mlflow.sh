#!/bin/sh
set -eu

MLFLOW_DATA_DIR="${MLFLOW_DATA_DIR:-/tmp/mlflow}"
MLFLOW_ARTIFACT_DIR="${MLFLOW_ARTIFACT_DIR:-$MLFLOW_DATA_DIR/artifacts}"
mkdir -p "$MLFLOW_ARTIFACT_DIR"

# Seed a fresh ephemeral tracker with the released model evaluation metrics.
# This runs against the same SQLite store before the HTTP server starts, so
# Render restarts recreate a useful project run when its free-tier disk resets.
python scripts/seed-mlflow-on-start.py

exec mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI:-sqlite:///$MLFLOW_DATA_DIR/mlflow.db}" \
  --artifacts-destination "$MLFLOW_ARTIFACT_DIR" \
  --serve-artifacts \
  --host 0.0.0.0 \
  --workers "${MLFLOW_WORKERS:-1}" \
  --port "${PORT:-5000}"
