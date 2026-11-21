"""Run the project's normal training pipeline with DagsHub MLflow tracking."""
from __future__ import annotations

import sys
from pathlib import Path

import dagshub


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Configure MLflow's tracking URI and authentication for this project. DagsHub
# uses its short-lived OAuth flow when this machine has no cached login.
dagshub.init(
    repo_owner="Mohan-mlk",
    repo_name="ad-performance-mlops",
    root=str(ROOT),
    mlflow=True,
)

from adperf.models.train import train  # noqa: E402


if __name__ == "__main__":
    train()
