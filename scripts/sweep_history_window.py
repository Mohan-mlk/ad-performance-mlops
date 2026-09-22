"""Pick the history-window length on business metrics, not on RMSE.

Each setting is a tracked MLflow run, so the choice is auditable later.
"""
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

results = []
for window in [8, 12, 26, 52, 200]:
    env = {**os.environ, "ADPERF_HISTORY_WINDOW": str(window), "PYTHONPATH": os.path.join(ROOT, "src")}
    subprocess.run([sys.executable, "-m", "adperf.features.build"], env=env, cwd=ROOT,
                   check=True, capture_output=True)
    subprocess.run([sys.executable, "-m", "adperf.models.train"], env=env, cwd=ROOT,
                   check=True, capture_output=True)
    with open(os.path.join(ROOT, "artifacts", "metrics.json")) as fh:
        m = json.load(fh)
    results.append({
        "window_weeks": window,
        "test_mae": round(m["test_mae"], 3),
        "spearman": round(m["test_spearman"], 3),
        "top20_lift": round(m["test_top20_lift"], 3),
        "auc": round(m.get("test_high_performer_auc", 0), 3),
        "realloc_uplift_pct": round(m["reallocation_uplift_pct"], 2),
    })
    print(results[-1], flush=True)

df = pd.DataFrame(results)
df["decision_score"] = df["spearman"].rank() + df["top20_lift"].rank() + df["realloc_uplift_pct"].rank()
print("\n" + df.sort_values("decision_score", ascending=False).to_string(index=False))
