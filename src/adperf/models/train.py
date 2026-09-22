"""Train the ROAS predictor and log everything to MLflow.

Why Tweedie: weekly ROAS is zero-inflated (roughly a quarter of ad-set weeks
bill nothing) but continuous and positive above zero. Squared error on that
distribution drags every prediction toward the middle and makes the model
useless for ranking. Tweedie with variance power ~1.5 is the standard fit for
that shape, and it is the same objective insurers use for claim cost.

The run is only accepted if it beats two baselines that cost nothing to deploy:
the account mean, and last-period category mean. A model that cannot beat
"just use the category average" has no business being served.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from adperf import config as C
from adperf.features import build as fb


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """Accuracy metrics plus the ranking metrics the media buyer actually uses."""
    rho = spearmanr(y_true, y_pred).statistic
    order = np.argsort(-y_pred)
    k = max(1, int(0.20 * len(y_true)))
    top_actual = y_true[order[:k]].mean()
    overall = y_true.mean()

    out = {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
        f"{prefix}spearman": float(rho),
        f"{prefix}top20_roas": float(top_actual),
        f"{prefix}top20_lift": float(top_actual / overall) if overall > 0 else 0.0,
    }
    hp_true = (y_true >= C.HIGH_PERFORMER_THRESHOLD).astype(int)
    if 0 < hp_true.sum() < len(hp_true):
        out[f"{prefix}high_performer_auc"] = float(roc_auc_score(hp_true, y_pred))
    return out


def reallocation_uplift(test: pd.DataFrame, y_pred: np.ndarray, move_share: float = 0.30) -> dict:
    """Simulated business impact: what if we had shifted budget by prediction?

    Take `move_share` of the spend sitting in the bottom-predicted quartile and
    move it to the top-predicted quartile, assuming the receiving ad sets hold
    their realised ROAS (conservative -- saturation would shave some of it).
    This is the number to put in front of the business, not RMSE.
    """
    d = test.copy()
    d["pred"] = y_pred
    lo_cut, hi_cut = d["pred"].quantile(0.25), d["pred"].quantile(0.75)
    lo, hi = d[d["pred"] <= lo_cut], d[d["pred"] >= hi_cut]
    if lo.empty or hi.empty or hi["spend_inr"].sum() == 0:
        return {"reallocation_uplift_pct": 0.0}

    moved = lo["spend_inr"].sum() * move_share
    lost = moved * (lo["revenue_inr"].sum() / max(lo["spend_inr"].sum(), 1.0))
    gained = moved * (hi["revenue_inr"].sum() / max(hi["spend_inr"].sum(), 1.0))
    base_rev = d["revenue_inr"].sum()
    return {
        "reallocation_moved_inr": float(moved),
        "reallocation_revenue_delta_inr": float(gained - lost),
        "reallocation_uplift_pct": float((gained - lost) / max(base_rev, 1.0) * 100),
    }


def train(cfg: C.TrainConfig | None = None, data_path: str | None = None) -> dict:
    import lightgbm as lgb

    cfg = cfg or C.TrainConfig()
    path = data_path or str(C.DATA_DIR / "processed" / "weekly.parquet")
    df = pd.read_parquet(path)
    for col in C.CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    fb.assert_no_leakage(C.FEATURES)
    train_df, valid_df, test_df = fb.time_split(df, cfg.valid_days, cfg.test_days)
    print(
        f"train={len(train_df):,} ({train_df.week.min().date()}..{train_df.week.max().date()})  "
        f"valid={len(valid_df):,}  test={len(test_df):,}"
    )

    X_tr, y_tr = train_df[C.FEATURES], train_df[C.TARGET]
    X_va, y_va = valid_df[C.FEATURES], valid_df[C.TARGET]
    X_te, y_te = test_df[C.FEATURES], test_df[C.TARGET]

    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=1,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.time()
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        categorical_feature=C.CATEGORICAL_FEATURES,
    )
    train_seconds = time.time() - t0

    pred_te = model.predict(X_te)
    metrics = _metrics(y_te.to_numpy(), pred_te, "test_")
    metrics.update(_metrics(y_va.to_numpy(), model.predict(X_va), "valid_"))
    metrics.update(reallocation_uplift(test_df, pred_te))
    metrics["train_seconds"] = train_seconds
    metrics["best_iteration"] = int(model.best_iteration_ or cfg.n_estimators)

    # ---- baselines it has to beat --------------------------------------
    mean_pred = np.full(len(y_te), y_tr.mean())
    cat_means = train_df.groupby("product_category", observed=True)[C.TARGET].mean()
    cat_pred = test_df["product_category"].map(cat_means).fillna(y_tr.mean()).to_numpy()
    metrics.update(_metrics(y_te.to_numpy(), mean_pred, "baseline_mean_"))
    metrics.update(_metrics(y_te.to_numpy(), cat_pred, "baseline_category_"))
    metrics["mae_improvement_vs_category_pct"] = float(
        (metrics["baseline_category_mae"] - metrics["test_mae"])
        / metrics["baseline_category_mae"]
        * 100
    )

    importance = (
        pd.DataFrame({"feature": C.FEATURES, "gain": model.booster_.feature_importance("gain")})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    importance["gain_share"] = importance["gain"] / importance["gain"].sum()

    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": C.FEATURES,
        "categorical_features": C.CATEGORICAL_FEATURES,
        "categories": {c: list(df[c].cat.categories) for c in C.CATEGORICAL_FEATURES},
        # The budget range the model actually saw. Trees cannot extrapolate, so
        # anything past this is a flat guess and the allocator must not trust it.
        "budget_support": {
            "p05": float(train_df["planned_budget_inr"].quantile(0.05)),
            "p50": float(train_df["planned_budget_inr"].median()),
            "p95": float(train_df["planned_budget_inr"].quantile(0.95)),
            "max": float(train_df["planned_budget_inr"].max()),
        },
        "trained_at": pd.Timestamp.now('UTC').isoformat(),
        "train_rows": len(train_df),
        "metrics": metrics,
        "target": C.TARGET,
    }
    joblib.dump(bundle, C.MODEL_PATH)
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    importance.to_csv(C.ARTIFACT_DIR / "feature_importance.csv", index=False)

    # Freeze the training window as the drift reference.
    C.REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(C.REFERENCE_PATH, index=False)

    _log_mlflow(model, cfg, metrics, importance, train_df)

    print("\n--- test metrics ---")
    for k in ["test_mae", "test_rmse", "test_r2", "test_spearman", "test_top20_lift",
              "test_high_performer_auc", "baseline_category_mae", "baseline_mean_mae",
              "mae_improvement_vs_category_pct", "reallocation_uplift_pct"]:
        if k in metrics:
            print(f"{k:38s} {metrics[k]:.4f}")
    print("\ntop features by gain:")
    print(importance.head(10).to_string(index=False))
    return metrics


def _log_mlflow(model, cfg, metrics, importance, train_df) -> None:
    """Track the run. Falls back to a no-op if MLflow is unavailable in CI."""
    try:
        import mlflow
        import mlflow.lightgbm
    except ImportError:  # pragma: no cover
        print("mlflow not installed - skipping tracking")
        return

    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"roas-tweedie-{pd.Timestamp.now('UTC'):%Y%m%d-%H%M%S}"):
        mlflow.set_tags({**cfg.tags, "objective": "tweedie", "grain": "adset_week"})
        mlflow.log_params(
            {
                "n_estimators": cfg.n_estimators,
                "learning_rate": cfg.learning_rate,
                "num_leaves": cfg.num_leaves,
                "min_child_samples": cfg.min_child_samples,
                "subsample": cfg.subsample,
                "colsample_bytree": cfg.colsample_bytree,
                "valid_days": cfg.valid_days,
                "test_days": cfg.test_days,
                "n_features": len(C.FEATURES),
                "train_rows": len(train_df),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(C.ARTIFACT_DIR / "feature_importance.csv"))
        mlflow.log_artifact(str(C.METRICS_PATH))
        try:
            mlflow.lightgbm.log_model(
                model, name="model", registered_model_name=C.REGISTERED_MODEL_NAME
            )
        except Exception as exc:  # noqa: BLE001 - registry may be unavailable; a
            # failed registration must never lose a finished training run
            print(f"model registry skipped: {exc}")
            mlflow.lightgbm.log_model(model, name="model")
    print(f"logged to MLflow at {C.MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    train()
