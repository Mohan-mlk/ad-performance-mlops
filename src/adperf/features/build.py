"""Turn raw ad-event rows into the modelling table.

Two design decisions carry most of the weight here:

1. **Weekly grain.** At ad x day granularity ~65% of rows have zero revenue,
   because a 2.8 lakh bridal set does not sell every day off a 4,800 rupee
   budget. Rolling up to ad set x week cuts that to ~26% and turns ROAS into a
   quantity worth regressing on.

2. **Pre-launch features only.** The model answers "what ROAS should I expect
   if I run this?", so it may not see impressions, clicks, CTR or conversions --
   those are outcomes. Campaign history enters only through expanding means
   that are shifted one week, so week *t* is scored using weeks < *t*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from adperf import config as C

GROUP_KEYS = [
    "campaign_id",
    "ad_id",
    "platform",
    "product_category",
    "creative_language",
    "creative_format",
    "audience_segment",
    "geo",
    "age_bracket",
    "gender_target",
    "bid_strategy",
]


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ad-event rows to one row per ad set per ISO week."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W").dt.start_time

    agg = (
        df.groupby(GROUP_KEYS + ["week"], observed=True)
        .agg(
            planned_budget_inr=("planned_budget_inr", "sum"),
            spend_inr=("spend_inr", "sum"),
            revenue_inr=("revenue_inr", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            leads=("leads", "sum"),
            conversions=("conversions", "sum"),
            days_to_festival=("days_to_festival", "min"),
            gold_price_per_gram=("gold_price_per_gram", "mean"),
            gold_price_momentum_30d=("gold_price_momentum_30d", "mean"),
            creative_age_days=("creative_age_days", "min"),
            audience_size_lakh=("audience_size_lakh", "first"),
            active_days=("date", "size"),
            weekend_share=("is_weekend", "mean"),
            payday_share=("is_payday_window", "mean"),
        )
        .reset_index()
    )

    # The festival label for the week = the one the week is closest to.
    fw = (
        df.sort_values("days_to_festival")
        .groupby(GROUP_KEYS + ["week"], observed=True)["festival_window"]
        .first()
        .reset_index()
    )
    agg = agg.merge(fw, on=GROUP_KEYS + ["week"], how="left")

    agg["week_of_year"] = agg["week"].dt.isocalendar().week.astype(int)
    agg["month"] = agg["week"].dt.month
    agg["ctr"] = np.where(agg["impressions"] > 0, agg["clicks"] / agg["impressions"], 0.0)
    agg[C.TARGET] = agg["revenue_inr"] / agg["spend_inr"].clip(lower=1.0)
    agg["is_high_performer"] = (agg[C.TARGET] >= C.HIGH_PERFORMER_THRESHOLD).astype(int)
    return agg.sort_values("week").reset_index(drop=True)


def _rolling_encoder(
    df: pd.DataFrame,
    keys: list[str],
    value_col: str,
    window: int | None = None,
    smoothing: float = 20.0,
) -> np.ndarray:
    """Smoothed trailing-window mean of `value_col` per key, using prior weeks only.

    A trailing window beats an expanding mean on two counts. It tracks recency,
    which matters when gold prices or creative trends move; and its variance
    does not shrink as history piles up, so the drift monitor is not fed a
    feature that converges by construction and looks like drift every quarter.

    Leakage control: the per-key series is shifted one week before the window is
    applied, so a row can never see its own week or anything after it.
    """
    window = window or C.HISTORY_WINDOW_WEEKS
    weekly = (
        df.groupby(keys + ["week"], observed=True)[value_col]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("week")
    )
    grp = weekly.groupby(keys, observed=True)
    prior_sum = grp["sum"].shift(1).rolling(window, min_periods=1).sum()
    prior_cnt = grp["count"].shift(1).rolling(window, min_periods=1).sum()

    # The smoothing prior must also come from the past. Using the full-sample
    # mean leaks the future into every cold-start row -- a subtle leak, but it
    # inflates offline metrics and then does not reproduce in production.
    prior_by_week = _trailing_global_prior(df, value_col)
    weekly = weekly.merge(prior_by_week, on="week", how="left")
    fallback = float(df.loc[df["week"] == df["week"].min(), value_col].mean())
    weekly["prior"] = weekly["prior"].fillna(fallback)

    weekly["enc"] = (prior_sum.fillna(0).to_numpy() + smoothing * weekly["prior"].to_numpy()) / (
        prior_cnt.fillna(0).to_numpy() + smoothing
    )

    out = df.merge(weekly[keys + ["week", "enc"]], on=keys + ["week"], how="left")
    return out["enc"].fillna(fallback).to_numpy()


def _trailing_global_prior(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Account-wide mean of `value_col` over all *earlier* weeks."""
    g = df.groupby("week", observed=True)[value_col].agg(["sum", "count"]).sort_index()
    prior = (g["sum"].cumsum() - g["sum"]) / (g["count"].cumsum() - g["count"]).replace(0, np.nan)
    return prior.rename("prior").reset_index()


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every encoder declared in `config.HISTORY_ENCODERS`."""
    df = df.sort_values("week").reset_index(drop=True)
    for name, (keys, value_col) in C.HISTORY_ENCODERS.items():
        df[name] = _rolling_encoder(df, keys, value_col)
    return df


def build(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full raw -> model-ready transform."""
    weekly = to_weekly(df_raw)
    weekly = add_history_features(weekly)
    for col in C.CATEGORICAL_FEATURES:
        weekly[col] = weekly[col].astype("category")
    return weekly


def time_split(df: pd.DataFrame, valid_days: int, test_days: int):
    """Chronological train / valid / test split. Never shuffle a time series."""
    weeks = df["week"]
    end = weeks.max()
    test_start = end - pd.Timedelta(days=test_days)
    valid_start = test_start - pd.Timedelta(days=valid_days)
    train = df[weeks < valid_start]
    valid = df[(weeks >= valid_start) & (weeks < test_start)]
    test = df[weeks >= test_start]
    return train, valid, test


def assert_no_leakage(feature_cols: list[str]) -> None:
    """Fail loudly if an outcome column ever sneaks into the feature list."""
    leaked = sorted(set(feature_cols) & set(C.LEAKY_COLUMNS))
    if leaked:
        raise ValueError(f"post-hoc outcome columns used as features: {leaked}")


def main() -> None:
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="Build the weekly modelling table")
    p.add_argument("--inp", default=str(C.RAW_PATH))
    p.add_argument("--out", default=str(C.DATA_DIR / "processed" / "weekly.parquet"))
    args = p.parse_args()

    raw = pd.read_parquet(args.inp)
    weekly = build(raw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(args.out, index=False)
    print(f"{len(raw):,} daily rows -> {len(weekly):,} weekly rows")
    print(f"zero-revenue weeks: {(weekly['revenue_inr'] == 0).mean():.1%}")
    print(f"median ROAS: {weekly[C.TARGET].median():.2f}x")


if __name__ == "__main__":
    main()
