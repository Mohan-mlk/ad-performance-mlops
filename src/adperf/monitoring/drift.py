"""Drift and performance monitoring.

Three different things can break a deployed ad model, and they need separate
alarms:

* **Feature drift** - the media plan changed (new geo, budgets doubled). PSI on
  numerics, and on category shares for the discrete columns.
* **Target drift** - ROAS itself moved, e.g. gold prices ran up. The model can
  be perfectly calibrated on last quarter and still be wrong.
* **Performance decay** - the only one that directly costs money. Needs labels,
  so it lags by the conversion window.

Feature drift alone is not a reason to retrain. Feature drift *plus* decay is.
That rule is encoded in `verdict()` so the on-call engineer is not guessing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from adperf import config as C

# Standard PSI reading: <0.10 stable, 0.10-0.25 watch, >0.25 material shift.
PSI_WATCH = 0.10
PSI_ALERT = 0.25


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index over quantile bins of the reference."""
    ref, cur = reference.dropna(), current.dropna()
    if len(ref) < 20 or len(cur) < 20:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6  # keeps an empty bucket from producing an infinite PSI
    ref_pct, cur_pct = np.clip(ref_pct, eps, None), np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI over category shares."""
    ref = reference.astype(str).value_counts(normalize=True)
    cur = current.astype(str).value_counts(normalize=True)
    levels = sorted(set(ref.index) | set(cur.index))
    eps = 1e-6
    r = np.clip(np.array([ref.get(l, 0.0) for l in levels]), eps, None)
    c = np.clip(np.array([cur.get(l, 0.0) for l in levels]), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Per-feature drift report, sorted worst first."""
    rows = []
    for col in C.FEATURES:
        if col not in reference or col not in current:
            continue
        if col in C.CATEGORICAL_FEATURES:
            psi = psi_categorical(reference[col], current[col])
            p_value = np.nan
        else:
            psi = psi_numeric(reference[col], current[col])
            p_value = float(ks_2samp(reference[col].dropna(), current[col].dropna()).pvalue)
        if col in C.CALENDAR_FEATURES:
            status = "SEASONAL"  # reported for context, never escalated
        elif psi > PSI_ALERT:
            status = "ALERT"
        elif psi > PSI_WATCH:
            status = "WATCH"
        else:
            status = "STABLE"
        rows.append(
            {
                "feature": col,
                "type": "categorical" if col in C.CATEGORICAL_FEATURES else "numeric",
                "psi": round(psi, 4),
                "ks_p_value": p_value,
                "status": status,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def season_matched_reference(
    reference: pd.DataFrame, current: pd.DataFrame, tolerance_days: int = 21
) -> pd.DataFrame:
    """Pick the comparable slice of history: the same calendar window, a year back.

    Comparing a six-week live window against thirty months of training data
    guarantees a false alarm on every seasonal column. Matching the season
    isolates genuine shift. Falls back to a same-length recent tail of the
    reference when no year-ago window exists.
    """
    lo, hi = current["week"].min(), current["week"].max()
    for years in (1, 2):
        start = lo - pd.DateOffset(years=years) - pd.Timedelta(days=tolerance_days)
        end = hi - pd.DateOffset(years=years) + pd.Timedelta(days=tolerance_days)
        window = reference[(reference["week"] >= start) & (reference["week"] <= end)]
        if len(window) >= 200:
            return window
    span = (hi - lo).days or 45
    return reference[reference["week"] >= reference["week"].max() - pd.Timedelta(days=span)]


def target_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    ref_y, cur_y = reference[C.TARGET], current[C.TARGET]
    return {
        "reference_mean_roas": round(float(ref_y.mean()), 3),
        "current_mean_roas": round(float(cur_y.mean()), 3),
        "mean_shift_pct": round(float((cur_y.mean() / max(ref_y.mean(), 1e-6) - 1) * 100), 2),
        "psi": round(psi_numeric(ref_y, cur_y), 4),
        "ks_p_value": float(ks_2samp(ref_y, cur_y).pvalue),
    }


def performance_decay(bundle: dict, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compare live error against the error the model was accepted with."""
    from sklearn.metrics import mean_absolute_error

    def _pred(frame: pd.DataFrame) -> np.ndarray:
        X = frame[bundle["features"]].copy()
        for col, cats in bundle["categories"].items():
            X[col] = pd.Categorical(X[col], categories=cats)
        return bundle["model"].predict(X)

    baseline_mae = bundle["metrics"].get("test_mae", mean_absolute_error(reference[C.TARGET], _pred(reference)))
    live_mae = float(mean_absolute_error(current[C.TARGET], _pred(current)))
    return {
        "baseline_mae": round(float(baseline_mae), 4),
        "live_mae": round(live_mae, 4),
        "mae_degradation_pct": round((live_mae / max(baseline_mae, 1e-6) - 1) * 100, 2),
    }


def verdict(drift_df: pd.DataFrame, target: dict, perf: dict) -> dict:
    """Decide what to do. Retraining is triggered by cost, not by curiosity."""
    alerts = drift_df[drift_df["status"] == "ALERT"]["feature"].tolist()
    watch = drift_df[drift_df["status"] == "WATCH"]["feature"].tolist()
    decayed = perf["mae_degradation_pct"] > 15
    target_moved = abs(target["mean_shift_pct"]) > 20

    if decayed and (alerts or target_moved):
        action, reason = "RETRAIN_NOW", "error is up materially and the inputs or the target have moved"
    elif decayed:
        action, reason = "INVESTIGATE", "error is up without an obvious drift cause - check tracking and attribution first"
    elif alerts:
        action, reason = "RETRAIN_SOON", "inputs have shifted materially; error has not degraded yet but it will"
    elif watch:
        action, reason = "MONITOR", "early drift signal, still within tolerance"
    else:
        action, reason = "HEALTHY", "no material drift and error is within tolerance"

    return {
        "action": action,
        "reason": reason,
        "alert_features": alerts,
        "watch_features": watch,
        "checked_at": pd.Timestamp.now("UTC").isoformat(),
    }


def run_report(bundle: dict, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    drift_df = feature_drift(reference, current)
    target = target_drift(reference, current)
    perf = performance_decay(bundle, reference, current)
    return {
        "feature_drift": drift_df.to_dict("records"),
        "target_drift": target,
        "performance": perf,
        "verdict": verdict(drift_df, target, perf),
        "reference_rows": len(reference),
        "current_rows": len(current),
    }


def main() -> None:
    import argparse

    import joblib

    p = argparse.ArgumentParser(description="Run the drift report")
    p.add_argument("--current-days", type=int, default=45, help="size of the live window")
    args = p.parse_args()

    bundle = joblib.load(C.MODEL_PATH)
    reference = pd.read_parquet(C.REFERENCE_PATH)
    weekly = pd.read_parquet(C.DATA_DIR / "processed" / "weekly.parquet")
    cutoff = weekly["week"].max() - pd.Timedelta(days=args.current_days)
    current = weekly[weekly["week"] >= cutoff]
    reference = season_matched_reference(reference, current)

    report = run_report(bundle, reference, current)
    out = C.ARTIFACT_DIR / "drift_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    print(f"verdict: {report['verdict']['action']} - {report['verdict']['reason']}")
    print(f"live MAE {report['performance']['live_mae']:.3f} vs baseline "
          f"{report['performance']['baseline_mae']:.3f} "
          f"({report['performance']['mae_degradation_pct']:+.1f}%)")
    print(f"target mean ROAS shift: {report['target_drift']['mean_shift_pct']:+.1f}%")
    print("\ntop drifting features:")
    print(pd.DataFrame(report["feature_drift"]).head(8).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
