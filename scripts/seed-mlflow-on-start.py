"""Create one transparent baseline run in a fresh hosted MLflow tracker."""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def main() -> None:
    data_dir = Path(os.getenv("MLFLOW_DATA_DIR", "/tmp/mlflow"))
    data_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "asr-ad-performance")
    metrics_path = Path(os.getenv("MLFLOW_METRICS_PATH", "/app/metrics.json"))

    health_url = f"{tracking_uri.rstrip('/')}/health"
    deadline = time.monotonic() + 90
    while True:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"MLflow server did not become healthy: {health_url}")
            time.sleep(1)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    if client.search_runs([experiment_id], max_results=1):
        return

    with metrics_path.open(encoding="utf-8") as stream:
        raw_metrics = json.load(stream)
    metrics = {
        str(name): float(value)
        for name, value in raw_metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    }

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="released-model-evaluation",
        tags={
            "source": "release-bundle-metrics",
            "model": "ad-performance-predictor",
            "dataset": "synthetic-project-benchmark",
            "description": "Evaluation metrics copied from the model release bundle at tracker startup.",
        },
    ):
        mlflow.log_params(
            {
                "run_type": "deployment_seed",
                "metric_source": "artifacts/metrics.json",
            }
        )
        mlflow.log_metrics(metrics)

    print(
        f"Seeded MLflow experiment {experiment_name!r} with "
        f"{len(metrics)} release evaluation metrics."
    )


if __name__ == "__main__":
    main()
