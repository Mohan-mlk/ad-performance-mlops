"""Rolling-origin backtest.

A single hold-out window of six weeks is not enough to tell two configurations
apart -- the fold-to-fold spread is wider than the difference between them. This
walks the origin forward, retraining at each step on everything before it, and
reports mean +/- std so a claimed improvement has to clear the noise floor.

This is the number that belongs in a report. The single-split metrics in
`train.py` are for the CI gate, where speed matters more.
"""
from __future__ import annotations

import json

import pandas as pd

from adperf import config as C
from adperf.models.train import _metrics, reallocation_uplift


def backtest(df: pd.DataFrame, n_folds: int = 6, horizon_days: int = 45, cfg=None) -> pd.DataFrame:
    """Walk the train/test origin forward `n_folds` times."""
    import lightgbm as lgb

    cfg = cfg or C.TrainConfig()
    for col in C.CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    end = df["week"].max()
    rows = []
    for fold in range(n_folds):
        test_end = end - pd.Timedelta(days=horizon_days * fold)
        test_start = test_end - pd.Timedelta(days=horizon_days)
        valid_start = test_start - pd.Timedelta(days=cfg.valid_days)

        train_df = df[df["week"] < valid_start]
        valid_df = df[(df["week"] >= valid_start) & (df["week"] < test_start)]
        test_df = df[(df["week"] >= test_start) & (df["week"] < test_end)]
        if len(train_df) < 500 or len(test_df) < 80 or len(valid_df) < 50:
            continue

        model = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            subsample_freq=1,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            random_state=cfg.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            train_df[C.FEATURES],
            train_df[C.TARGET],
            eval_set=[(valid_df[C.FEATURES], valid_df[C.TARGET])],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
            categorical_feature=C.CATEGORICAL_FEATURES,
        )
        pred = model.predict(test_df[C.FEATURES])
        y = test_df[C.TARGET].to_numpy()

        cat_means = train_df.groupby("product_category", observed=True)[C.TARGET].mean()
        cat_pred = test_df["product_category"].map(cat_means).fillna(train_df[C.TARGET].mean()).to_numpy()

        m = _metrics(y, pred)
        m.update({f"baseline_{k}": v for k, v in _metrics(y, cat_pred).items()})
        m.update(reallocation_uplift(test_df, pred))
        m.update(
            {
                "fold": fold,
                "test_start": test_start.date().isoformat(),
                "test_rows": len(test_df),
                "train_rows": len(train_df),
            }
        )
        rows.append(m)
        print(f"fold {fold}: test from {test_start.date()} "
              f"mae={m['mae']:.3f} spearman={m['spearman']:.3f} top20_lift={m['top20_lift']:.2f}")

    return pd.DataFrame(rows)


def summarise(folds: pd.DataFrame) -> dict:
    cols = ["mae", "rmse", "spearman", "top20_lift", "high_performer_auc",
            "baseline_mae", "reallocation_uplift_pct"]
    out = {}
    for c in cols:
        if c in folds:
            out[f"{c}_mean"] = round(float(folds[c].mean()), 4)
            out[f"{c}_std"] = round(float(folds[c].std(ddof=1)), 4)
    out["n_folds"] = len(folds)
    out["mae_improvement_vs_category_pct"] = round(
        float((folds["baseline_mae"].mean() - folds["mae"].mean()) / folds["baseline_mae"].mean() * 100), 2
    )
    # Paired across folds: does the model beat the baseline every single time?
    out["folds_beating_baseline"] = int((folds["mae"] < folds["baseline_mae"]).sum())
    return out


def main() -> None:
    df = pd.read_parquet(C.DATA_DIR / "processed" / "weekly.parquet")
    folds = backtest(df)
    summary = summarise(folds)
    folds.to_csv(C.ARTIFACT_DIR / "backtest_folds.csv", index=False)
    (C.ARTIFACT_DIR / "backtest_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- rolling-origin summary ---")
    for k, v in summary.items():
        print(f"{k:42s} {v}")


if __name__ == "__main__":
    main()
