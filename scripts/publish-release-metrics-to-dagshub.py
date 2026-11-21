"""Publish the bundled release metrics to this project's DagsHub MLflow tracker.

Authentication uses DagsHub's short-lived OAuth flow. The temporary token
cache is deleted as soon as the upload completes and is never committed.
"""
from __future__ import annotations

import base64
import json
import math
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import dagshub.auth


TRACKING_URI = "https://dagshub.com/Mohan-mlk/ad-performance-mlops.mlflow"
EXPERIMENT_NAME = "asr-ad-performance"
METRICS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "metrics.json"


def mlflow_request(token: str, endpoint: str, payload: dict) -> dict:
    credentials = base64.b64encode(f"Mohan-mlk:{token}".encode()).decode("ascii")
    request = urllib.request.Request(
        f"{TRACKING_URI}/api/2.0/mlflow/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DagsHub MLflow request failed ({exc.code}): {detail}") from exc


def main() -> None:
    with METRICS_PATH.open(encoding="utf-8") as stream:
        source = json.load(stream)
    metrics = {
        str(key): float(value)
        for key, value in source.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    }

    # Use DagsHub's short-lived OAuth flow. The temporary cache is removed as
    # soon as the upload completes, so no token remains on disk.
    with tempfile.TemporaryDirectory(prefix="dagshub-mlflow-auth-") as cache_dir:
        cache_path = Path(cache_dir) / "tokens.json"
        token = dagshub.auth.get_token(cache_location=str(cache_path))
        result = mlflow_request(
            token,
            "experiments/get-by-name",
            {"experiment_name": EXPERIMENT_NAME},
        )
        experiment_id = result["experiment"]["experiment_id"]
        started_ms = int(time.time() * 1000)
        created = mlflow_request(
            token,
            "runs/create",
            {
                "experiment_id": experiment_id,
                "start_time": started_ms,
                "run_name": "released-model-evaluation",
                "tags": [
                    {"key": "source", "value": "release-bundle-metrics"},
                    {"key": "model", "value": "ad-performance-predictor"},
                    {"key": "dataset", "value": "synthetic-project-benchmark"},
                    {"key": "run_type", "value": "release_evaluation"},
                ],
            },
        )
        run_id = created["run"]["info"]["run_id"]
        mlflow_request(
            token,
            "runs/log-batch",
            {
                "run_id": run_id,
                "metrics": [
                    {
                        "key": key,
                        "value": value,
                        "timestamp": started_ms,
                        "step": 0,
                    }
                    for key, value in metrics.items()
                ],
                "params": [
                    {"key": "metric_source", "value": "artifacts/metrics.json"},
                    {"key": "data_type", "value": "synthetic benchmark"},
                ],
            },
        )
        mlflow_request(
            token,
            "runs/update",
            {"run_id": run_id, "status": "FINISHED", "end_time": int(time.time() * 1000)},
        )
    print(f"Logged {len(metrics)} release metrics to DagsHub MLflow.")
    print(f"Run: {TRACKING_URI}/#/experiments/{experiment_id}/runs/{run_id}")


if __name__ == "__main__":
    main()
