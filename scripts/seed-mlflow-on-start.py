"""Create one transparent baseline run in a fresh hosted MLflow tracker."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def main() -> None:
    data_dir = Path(os.getenv("MLFLOW_DATA_DIR", "/tmp/mlflow"))
    data_dir.mkdir(parents=True, exist_ok=True)
    backend_uri = os.getenv(
        "MLFLOW_BACKEND_STORE_URI", f"sqlite:///{data_dir / 'mlflow.db'}"
    )
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "asr-ad-performance")
    metrics_path = Path(os.getenv("MLFLOW_METRICS_PATH", "/app/metrics.json"))

    mlflow.set_tracking_uri(backend_uri)
    client = MlflowClient(tracking_uri=backend_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        # Artifacts are proxied by the serving MLflow process. This seeded run
        # logs evaluation metrics only; normal training runs can log artifacts.
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location="mlflow-artifacts:/1",
        )
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


if __name__ == "__main__":
    main()
