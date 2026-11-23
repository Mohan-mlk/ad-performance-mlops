"""Seed a hosted MLflow tracker with the project's release and data lineage."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]


def _log_dataset(
    frame: pd.DataFrame,
    *,
    name: str,
    source: str,
    context: str,
    source_sha256: str,
) -> None:
    """Record a dataset's schema/profile in MLflow without exposing real data."""
    from adperf import config as C

    # MLflow 2.10's PandasDataset profile can be very slow on the full raw
    # event table. Use a deterministic, evenly spaced sample for the UI schema
    # and profile; the complete parquet is uploaded below as a run artifact.
    full_row_count = len(frame)
    sample_stride = max(1, (full_row_count + 999) // 1000)
    logged_frame = frame.iloc[::sample_stride].copy()
    for column in C.CATEGORICAL_FEATURES:
        if column in logged_frame:
            logged_frame[column] = logged_frame[column].astype("string").fillna("")
    dataset = mlflow.data.from_pandas(
        logged_frame,
        source=source,
        name=name,
        # Supplying the parquet's content digest avoids MLflow's default
        # dataframe-wide type scan, which can take minutes on this service.
        digest=source_sha256[:32],
        targets=C.TARGET if C.TARGET in logged_frame else None,
    )
    mlflow.log_input(
        dataset,
        context=context,
        tags={
            "source_sha256": source_sha256,
            "dataset_is_synthetic": "true",
            "source_row_count": str(full_row_count),
            "metadata_sample_rows": str(len(logged_frame)),
        },
    )


def main() -> None:
    data_dir = Path(os.getenv("MLFLOW_DATA_DIR", "/tmp/mlflow"))
    data_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "asr-ad-performance")
    metrics_path = Path(os.getenv("MLFLOW_METRICS_PATH", str(ROOT / "artifacts" / "metrics.json")))
    raw_path = ROOT / "data" / "raw" / "ad_events.parquet"
    weekly_path = ROOT / "data" / "processed" / "weekly.parquet"
    params_path = ROOT / "params.yaml"

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

    # A prior release-only seed run had no MLflow dataset inputs or project
    # artifacts. Seed the richer run once, even when that older run exists.
    existing_project_run = client.search_runs(
        [experiment_id],
        filter_string="tags.project_integration_version = '4'",
        max_results=1,
    )
    if existing_project_run:
        return

    with metrics_path.open(encoding="utf-8") as stream:
        raw_metrics = json.load(stream)
    metrics = {
        str(name): float(value)
        for name, value in raw_metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    }

    from adperf import config as C
    from adperf.features import build as feature_builder

    with params_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    raw = pd.read_parquet(raw_path)
    weekly = pd.read_parquet(weekly_path)
    for column in C.CATEGORICAL_FEATURES:
        weekly[column] = weekly[column].astype("category")
    train_df, valid_df, test_df = feature_builder.time_split(
        weekly,
        valid_days=int(config["train"]["valid_days"]),
        test_days=int(config["train"]["test_days"]),
    )
    params = {
        "run_type": "project_release_lineage",
        "source_revision": os.getenv("RENDER_GIT_COMMIT", "unknown"),
        "dataset_rows": len(raw),
        "weekly_rows": len(weekly),
        "data_seed": config["data"]["seed"],
        "data_rows_configured": config["data"]["rows"],
        "history_window_weeks": config["features"]["history_window_weeks"],
        **{f"train_{key}": value for key, value in config["train"].items()},
    }
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    weekly_sha256 = hashlib.sha256(weekly_path.read_bytes()).hexdigest()

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="project-release-with-datasets-v4",
        tags={
            "source": "project-release-bundle",
            "model": "ad-performance-predictor",
            "dataset": "synthetic-ad-events-and-weekly-features",
            "project_integration_version": "4",
            "description": "Release evaluation with the project's actual synthetic data, time splits, model, and monitoring artifacts.",
        },
    ):
        mlflow.log_params({str(key): str(value) for key, value in params.items()})
        mlflow.log_metrics(metrics)

        _log_dataset(
            raw,
            name="ad-performance-raw-ad-events",
            source=raw_path.as_uri(),
            context="source",
            source_sha256=raw_sha256,
        )
        for context, split_name, frame in (
            ("training", "train", train_df),
            ("validation", "validation", valid_df),
            ("testing", "test", test_df),
        ):
            _log_dataset(
                frame[[*C.FEATURES, C.TARGET]],
                name=f"ad-performance-weekly-{split_name}",
                source=weekly_path.as_uri(),
                context=context,
                source_sha256=weekly_sha256,
            )

        # Upload the actual synthetic data files as downloadable artifacts in
        # addition to registering their schemas as MLflow dataset inputs.
        mlflow.log_artifact(str(raw_path), artifact_path="datasets/source")
        mlflow.log_artifact(str(weekly_path), artifact_path="datasets/weekly")

        artifact_names = (
            "model.joblib",
            "metrics.json",
            "feature_importance.csv",
            "insights.json",
            "drift_report.json",
            "backtest_summary.json",
        )
        for artifact_name in artifact_names:
            artifact_path = ROOT / "artifacts" / artifact_name
            if artifact_path.is_file():
                mlflow.log_artifact(str(artifact_path), artifact_path="project-release")
        reference_path = ROOT / "data" / "reference" / "reference_window.parquet"
        if reference_path.is_file():
            mlflow.log_artifact(str(reference_path), artifact_path="project-release/reference")

        manifest = {
            "project": "ad-performance-mlops",
            "experiment": experiment_name,
            "dataset": {
                "raw_path": "data/raw/ad_events.parquet",
                "raw_rows": len(raw),
                "raw_sha256": raw_sha256,
                "weekly_path": "data/processed/weekly.parquet",
                "weekly_rows": len(weekly),
                "weekly_sha256": weekly_sha256,
                "target": C.TARGET,
                "features": C.FEATURES,
                "split_rows": {
                    "training": len(train_df),
                    "validation": len(valid_df),
                    "testing": len(test_df),
                },
                "data_is_synthetic": True,
            },
            "artifacts": [
                name for name in artifact_names
                if (ROOT / "artifacts" / name).is_file()
            ] + (["data/reference/reference_window.parquet"] if reference_path.is_file() else []),
            "tracking_uri": tracking_uri,
        }
        mlflow.log_dict(manifest, "lineage/project_run_manifest.json")

    print(
        f"Seeded MLflow experiment {experiment_name!r} with release metrics, "
        f"{len(raw):,} raw rows, {len(weekly):,} weekly rows, and project artifacts."
    )


if __name__ == "__main__":
    main()
