"""Seed a fresh hosted tracker with the model lineage in the release bundle."""
from __future__ import annotations

import math
import os
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"http://127.0.0.1:{os.getenv('PORT', '5000')}"
)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "asr-ad-performance")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    if client.search_runs([experiment_id], max_results=1):
        print(f"MLflow experiment {EXPERIMENT_NAME!r} already has a run; seed skipped.")
        return

    bundle = joblib.load(MODEL_PATH)
    lineage = bundle.get("mlflow", {})
    metrics = bundle.get("metrics", {})
    metric_values = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    }
    params = lineage.get("params", {})
    params = {str(key): str(value) for key, value in params.items()}

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="deployed-model-lineage",
        tags={
            "source": "project-release-bundle",
            "source_run_id": lineage.get("run_id", "unavailable"),
            "source_revision": lineage.get("source_revision", "unknown"),
            "description": "Lineage and metrics copied from the model already trained for this release.",
        },
    ):
        if params:
            mlflow.log_params(params)
        if metric_values:
            mlflow.log_metrics(metric_values)
        mlflow.log_artifact(str(MODEL_PATH))
        if METRICS_PATH.exists():
            mlflow.log_artifact(str(METRICS_PATH))

    print(f"Seeded MLflow experiment {EXPERIMENT_NAME!r} from the deployed model bundle.")


if __name__ == "__main__":
    main()
