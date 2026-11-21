"""Train the ad ROAS model and log its run to the hosted Render MLflow UI."""
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ["MLFLOW_TRACKING_URI"] = "https://asr-adperf-mlflow.onrender.com"
os.environ.setdefault("MLFLOW_EXPERIMENT", "asr-ad-performance")
sys.path.insert(0, str(ROOT / "src"))

from adperf.models.train import train  # noqa: E402


if __name__ == "__main__":
    train()
