"""Contract tests for the serving layer.

These run against the real model bundle, so they double as a deployment smoke
test: if the artifact and the code have drifted apart, these fail.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from adperf import config as C

pytestmark = pytest.mark.skipif(not C.MODEL_PATH.exists(), reason="no trained model available")


@pytest.fixture(scope="module")
def client():
    from adperf.serving.api import app

    with TestClient(app) as c:
        yield c


BASE = {
    "product_category": "bridal_set",
    "creative_language": "telugu",
    "geo": "vijayawada",
    "creative_format": "reel",
    "platform": "meta_instagram",
    "audience_segment": "bridal",
    "planned_budget_inr": 35_000,
    "days_to_festival": 12,
    "festival_window": "wedding_season_peak",
}


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["status"] == "ready"


def test_model_info_exposes_metrics(client):
    info = client.get("/model-info").json()
    assert info["target"] == "roas"
    assert info["n_features"] == len(C.FEATURES)


def test_predict_returns_a_usable_answer(client):
    body = client.post("/predict", json=BASE).json()
    assert body["predicted_roas"] > 0
    assert body["predicted_revenue_inr"] == pytest.approx(
        body["predicted_roas"] * BASE["planned_budget_inr"], rel=1e-3
    )
    assert body["verdict"]


def test_history_features_are_optional(client):
    """A brand-new campaign has no history; the service must still answer."""
    minimal = {"product_category": "gold_coin", "planned_budget_inr": 10_000}
    assert client.post("/predict", json=minimal).status_code == 200


def test_unseen_category_level_is_rejected_not_guessed(client):
    bad = {**BASE, "geo": "chennai"}
    assert client.post("/predict", json=bad).status_code == 422


def test_invalid_budget_is_rejected(client):
    assert client.post("/predict", json={**BASE, "planned_budget_inr": -5}).status_code == 422


def test_batch_ranks_descending(client):
    plans = [BASE, {**BASE, "creative_language": "english"}, {**BASE, "geo": "hyderabad"}]
    body = client.post("/predict/batch", json={"plans": plans}).json()
    roas = [r["predicted_roas"] for r in body["results"]]
    assert roas == sorted(roas, reverse=True)
    assert body["count"] == 3


def test_allocation_respects_the_budget(client):
    plans = [BASE, {**BASE, "product_category": "gold_coin"}, {**BASE, "geo": "guntur"}]
    body = client.post(
        "/allocate", json={"plans": plans, "total_budget_inr": 150_000}
    ).json()
    spent = sum(a["allocated_budget_inr"] for a in body["allocation"])
    assert spent <= 150_000 + 1e-6
    assert body["uplift_pct"] >= 0  # optimiser never loses to an even split
