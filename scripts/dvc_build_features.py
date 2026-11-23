"""DVC entry point for weekly feature generation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
os.environ["ADPERF_HISTORY_WINDOW"] = str(params["features"]["history_window_weeks"])
sys.path.insert(0, str(ROOT / "src"))

from adperf.features.build import main  # noqa: E402


if __name__ == "__main__":
    main()
