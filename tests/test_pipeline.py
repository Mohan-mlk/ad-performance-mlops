"""Tests for the ASR ad-performance pipeline.

Several of these exist because the bug actually happened during development:
the ROAS sanity test caught a generator that produced 505x returns, and the
seasonal-drift test caught a monitor that alerted every week simply because the
calendar had moved. Tests that encode real failures are worth more than tests
that restate the implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adperf import config as C
from adperf.data import generate as gen
from adperf.features import build as fb
from adperf.monitoring import drift as dr


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return gen.generate(n_rows=6_000, seed=11)


@pytest.fixture(scope="module")
def weekly(raw: pd.DataFrame) -> pd.DataFrame:
    return fb.build(raw)


# ------------------------------------------------------------------ data
def test_schema_and_no_nulls(raw: pd.DataFrame) -> None:
    required = set(C.CATEGORICAL_FEATURES) - {"festival_window"} | {"date", "spend_inr", "revenue_inr"}
    assert required.issubset(raw.columns)
    assert not raw[list(required)].isna().any().any()


def test_funnel_is_monotone(raw: pd.DataFrame) -> None:
    """Impressions >= clicks >= leads >= conversions, always."""
    assert (raw["impressions"] >= raw["clicks"]).all()
    assert (raw["clicks"] >= raw["leads"]).all()
    assert (raw["leads"] >= raw["conversions"]).all()


def test_roas_is_plausible_for_jewellery(raw: pd.DataFrame) -> None:
    """Guards the bug where clicks converted straight into 2.8 lakh purchases.

    Without the enquiry -> showroom -> sale funnel, blended ROAS came out at
    505x. Anything above ~25x blended means the close-rate step is broken.
    """
    blended = raw["revenue_inr"].sum() / raw["spend_inr"].sum()
    assert 2 < blended < 25, f"implausible blended ROAS: {blended:.1f}x"


def test_adsets_are_persistent(raw: pd.DataFrame) -> None:
    """Ad sets must deliver over a flight, not appear for a single day."""
    days_per_ad = raw.groupby("ad_id")["date"].nunique()
    assert days_per_ad.median() >= 5
    # An ad set keeps its settings for its whole flight.
    assert (raw.groupby("ad_id")["product_category"].nunique() == 1).all()


def test_known_effects_are_present(raw: pd.DataFrame) -> None:
    """Telugu creative should beat English in Vijayawada -- the core hypothesis."""
    vij = raw[raw["geo"] == "vijayawada"]
    by_lang = vij.groupby("creative_language").apply(
        lambda d: d["revenue_inr"].sum() / max(d["spend_inr"].sum(), 1), include_groups=False
    )
    assert by_lang["telugu"] > by_lang["english"]


# -------------------------------------------------------------- features
def test_no_leaky_features_in_config() -> None:
    fb.assert_no_leakage(C.FEATURES)
    with pytest.raises(ValueError, match="post-hoc"):
        fb.assert_no_leakage(C.FEATURES + ["clicks"])


def test_history_encoder_never_sees_its_own_week() -> None:
    """The encoder must be computable from strictly earlier weeks.

    Constructed so the target jumps in the final week: if the encoder leaked,
    the last week's value would move with it. It must not.
    """
    weeks = pd.date_range("2025-01-06", periods=6, freq="7D")
    df = pd.DataFrame(
        {
            "week": np.repeat(weeks, 4),  # 4 rows per week, in week order
            "product_category": ["gold_coin"] * 24,
            "roas": [2.0] * 20 + [99.0] * 4,  # spike confined to the final week
        }
    )
    enc = fb._rolling_encoder(df, ["product_category"], "roas", smoothing=1.0)
    last = enc[df["week"] == weeks[-1]]
    assert last.max() < 10, "encoder leaked the current week into its own feature"
    # And the spike must show up the week *after* it happened, if there were one.
    assert np.isclose(last.max(), 2.0, atol=0.5)


def test_weekly_aggregation_preserves_money(raw: pd.DataFrame, weekly: pd.DataFrame) -> None:
    assert weekly["spend_inr"].sum() == pytest.approx(raw["spend_inr"].sum(), rel=1e-6)
    assert weekly["revenue_inr"].sum() == pytest.approx(raw["revenue_inr"].sum(), rel=1e-6)


def test_weekly_aggregation_reduces_zero_inflation(raw: pd.DataFrame, weekly: pd.DataFrame) -> None:
    daily_zero = (raw["revenue_inr"] == 0).mean()
    weekly_zero = (weekly["revenue_inr"] == 0).mean()
    assert weekly_zero < daily_zero


def test_time_split_is_chronological(weekly: pd.DataFrame) -> None:
    train, valid, test = fb.time_split(weekly, 45, 45)
    assert train["week"].max() < valid["week"].min()
    assert valid["week"].max() < test["week"].min()
    assert len(test) > 0


# -------------------------------------------------------------- monitoring
def test_psi_is_zero_for_identical_distributions() -> None:
    s = pd.Series(np.random.default_rng(0).normal(size=2_000))
    assert dr.psi_numeric(s, s.copy()) < 1e-6
    c = pd.Series(["a", "b", "c"] * 300)
    assert dr.psi_categorical(c, c.copy()) < 1e-6


def test_psi_detects_a_real_shift() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 3_000))
    cur = pd.Series(rng.normal(2.5, 1, 3_000))
    assert dr.psi_numeric(ref, cur) > dr.PSI_ALERT


def test_calendar_features_never_escalate(weekly: pd.DataFrame) -> None:
    """Seasonality is not drift.

    The first monitor built here alerted on `month` and `days_to_festival` every
    single run, because comparing two different time windows always moves them.
    Alerts nobody can act on get ignored, so these are reported, never escalated.
    """
    cutoff = weekly["week"].max() - pd.Timedelta(days=45)
    report = dr.feature_drift(weekly[weekly["week"] < cutoff], weekly[weekly["week"] >= cutoff])
    calendar_rows = report[report["feature"].isin(C.CALENDAR_FEATURES)]
    assert len(calendar_rows) > 0
    assert (calendar_rows["status"] == "SEASONAL").all()


def test_season_matched_reference_picks_a_year_ago(weekly: pd.DataFrame) -> None:
    cutoff = weekly["week"].max() - pd.Timedelta(days=45)
    current = weekly[weekly["week"] >= cutoff]
    ref = dr.season_matched_reference(weekly[weekly["week"] < cutoff], current)
    assert len(ref) > 0
    assert ref["week"].max() < current["week"].min()


def test_verdict_requires_cost_not_just_drift() -> None:
    """Feature drift alone must not trigger a retrain."""
    drifted = pd.DataFrame([{"feature": "geo", "status": "ALERT", "psi": 0.9}])
    stable_target = {"mean_shift_pct": 1.0}
    healthy_perf = {"mae_degradation_pct": 1.0}
    assert dr.verdict(drifted, stable_target, healthy_perf)["action"] == "RETRAIN_SOON"

    decayed = {"mae_degradation_pct": 40.0}
    assert dr.verdict(drifted, stable_target, decayed)["action"] == "RETRAIN_NOW"

    clean = pd.DataFrame([{"feature": "geo", "status": "STABLE", "psi": 0.01}])
    assert dr.verdict(clean, stable_target, healthy_perf)["action"] == "HEALTHY"
