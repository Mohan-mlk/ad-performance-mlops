"""Pull or push this project's DVC cache using DagsHub OAuth credentials."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import dagshub.auth


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pull", "push"))
    args = parser.parse_args()

    token = dagshub.auth.get_token()
    environment = os.environ.copy()
    environment["AWS_ACCESS_KEY_ID"] = token
    environment["AWS_SECRET_ACCESS_KEY"] = token
    environment["DVC_SITE_CACHE_DIR"] = str(ROOT / ".dvc-site-cache")
    subprocess.run(
        [sys.executable, "-m", "dvc", args.action, "-r", "dagshub"],
        cwd=ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
