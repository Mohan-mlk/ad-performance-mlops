"""DVC entry point for model training with tracked hyperparameters."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adperf import config as C  # noqa: E402
from adperf.models.train import train  # noqa: E402


if __name__ == "__main__":
    parameters = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    train(cfg=C.TrainConfig(**parameters["train"]))
