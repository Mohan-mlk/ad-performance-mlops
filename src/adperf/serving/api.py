"""FastAPI service for the ASR ad-performance model.

Endpoints are shaped around the questions a media buyer actually asks:
"should I run this?" (/predict), "rank these for me" (/predict/batch), "how do I
split the money?" (/allocate) and "what should I change?" (/insights).

Operational notes:
* The model bundle is loaded once at startup, not per request.
* `/health` is a liveness probe; `/ready` fails until the model is loaded, so a
  rolling deploy cannot send traffic to a pod with no model.
* History features are optional in the request. A buyer planning a brand-new
  campaign does not know last quarter's category ROAS, so the service fills
  them from the account defaults baked into the bundle at train time.
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from adperf import config as C

STATE: dict = {"bundle": None, "defaults": {}, "insights": None, "served": 0}


def _load() -> None:
    if not C.MODEL_PATH.exists():
        raise FileNotFoundError(f"no model at {C.MODEL_PATH} - run training first")
    bundle = joblib.load(C.MODEL_PATH)
    STATE["bundle"] = bundle

    # Account-level fallbacks for the history features.
    ref = pd.read_parquet(C.REFERENCE_PATH) if C.REFERENCE_PATH.exists() else None
    defaults = {}
    for col in C.NUMERIC_FEATURES:
        defaults[col] = float(ref[col].median()) if ref is not None and col in ref else 0.0
    STATE["defaults"] = defaults

    if C.INSIGHTS_PATH.exists():
        STATE["insights"] = json.loads(C.INSIGHTS_PATH.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield
    STATE.clear()


app = FastAPI(
    title="ASR Jewellery Works - Ad Performance API",
    description="Predicted ROAS and marketing insights for campaign planning.",
    version="1.0.0",
    lifespan=lifespan,
)


class AdPlan(BaseModel):
    """One planned ad set for one week."""

    platform: Literal[tuple(C.PLATFORMS)] = "meta_instagram"  # type: ignore[valid-type]
    product_category: Literal[tuple(C.PRODUCT_CATEGORIES)] = "gold_chain"  # type: ignore[valid-type]
    creative_language: Literal[tuple(C.CREATIVE_LANGUAGES)] = "telugu"  # type: ignore[valid-type]
    creative_format: Literal[tuple(C.CREATIVE_FORMATS)] = "reel"  # type: ignore[valid-type]
    audience_segment: Literal[tuple(C.AUDIENCE_SEGMENTS)] = "daily_wear"  # type: ignore[valid-type]
    geo: Literal[tuple(C.GEOS)] = "vijayawada"  # type: ignore[valid-type]
    age_bracket: Literal[tuple(C.AGE_BRACKETS)] = "25-34"  # type: ignore[valid-type]
    gender_target: Literal[tuple(C.GENDER_TARGETS)] = "female"  # type: ignore[valid-type]
    bid_strategy: Literal[tuple(C.BID_STRATEGIES)] = "target_roas"  # type: ignore[valid-type]
    festival_window: str = "none"

    planned_budget_inr: float = Field(20_000, gt=0, description="Weekly budget in INR")
    days_to_festival: int = Field(60, ge=0, le=60)
    gold_price_per_gram: float | None = None
    gold_price_momentum_30d: float | None = None
    creative_age_days: int = Field(0, ge=0)
    audience_size_lakh: float = Field(6.5, gt=0)
    active_days: int = Field(7, ge=1, le=7)
    week_of_year: int = Field(1, ge=1, le=53)
    month: int = Field(1, ge=1, le=12)
    weekend_share: float = Field(2 / 7, ge=0, le=1)
    payday_share: float = Field(0.3, ge=0, le=1)

    # Optional account history. Filled from training-set medians when absent.
    hist_roas_category: float | None = None
    hist_roas_platform_format: float | None = None
    hist_roas_campaign: float | None = None
    hist_ctr_creative_lang: float | None = None


class PredictResponse(BaseModel):
    predicted_roas: float
    predicted_revenue_inr: float
    verdict: str
    high_performer_probability_proxy: float
    latency_ms: float


class BatchRequest(BaseModel):
    plans: list[AdPlan] = Field(..., min_length=1, max_length=500)


class AllocateRequest(BaseModel):
    plans: list[AdPlan] = Field(..., min_length=2, max_length=60)
    total_budget_inr: float = Field(..., gt=0)
    min_per_adset_inr: float = Field(0, ge=0)
    max_per_adset_inr: float | None = None


def _to_frame(plans: list[AdPlan]) -> pd.DataFrame:
    rows = []
    for p in plans:
        d = p.model_dump()
        for col, fallback in STATE["defaults"].items():
            if d.get(col) is None:
                d[col] = fallback
        rows.append(d)
    return pd.DataFrame(rows)


def _predict(frame: pd.DataFrame) -> np.ndarray:
    bundle = STATE["bundle"]
    if bundle is None:
        raise HTTPException(503, "model not loaded")
    X = frame[bundle["features"]].copy()
    for col, cats in bundle["categories"].items():
        X[col] = pd.Categorical(X[col], categories=cats)
        if X[col].isna().any():
            bad = frame.loc[X[col].isna(), col].unique().tolist()
            raise HTTPException(422, f"unseen level(s) for {col}: {bad}")
    return np.asarray(bundle["model"].predict(X), dtype=float)


def _verdict(roas: float) -> str:
    if roas >= 8:
        return "SCALE - well above account average"
    if roas >= C.HIGH_PERFORMER_THRESHOLD:
        return "RUN - expected to clear the profitability bar"
    if roas >= 1.5:
        return "TEST SMALL - marginal, cap the budget"
    return "DO NOT RUN - predicted below break-even on media cost"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "predictions_served": STATE["served"]}


@app.get("/ready")
def ready() -> dict:
    if STATE.get("bundle") is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready"}


@app.get("/model-info")
def model_info() -> dict:
    b = STATE["bundle"]
    if b is None:
        raise HTTPException(503, "model not loaded")
    m = b["metrics"]
    return {
        "trained_at": b["trained_at"],
        "train_rows": b["train_rows"],
        "target": b["target"],
        "n_features": len(b["features"]),
        "test_mae": m.get("test_mae"),
        "test_spearman": m.get("test_spearman"),
        "top20_lift": m.get("test_top20_lift"),
        "mae_improvement_vs_category_pct": m.get("mae_improvement_vs_category_pct"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(plan: AdPlan) -> PredictResponse:
    t0 = time.perf_counter()
    roas = float(_predict(_to_frame([plan]))[0])
    STATE["served"] += 1
    return PredictResponse(
        predicted_roas=round(roas, 3),
        predicted_revenue_inr=round(roas * plan.planned_budget_inr, 2),
        verdict=_verdict(roas),
        # Not a calibrated probability -- a monotone proxy for ranking only.
        high_performer_probability_proxy=round(
            float(1 / (1 + np.exp(-(roas - C.HIGH_PERFORMER_THRESHOLD)))), 3
        ),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> dict:
    t0 = time.perf_counter()
    frame = _to_frame(req.plans)
    preds = _predict(frame)
    STATE["served"] += len(preds)
    results = [
        {
            "index": i,
            "product_category": req.plans[i].product_category,
            "geo": req.plans[i].geo,
            "creative_language": req.plans[i].creative_language,
            "predicted_roas": round(float(p), 3),
            "predicted_revenue_inr": round(float(p) * req.plans[i].planned_budget_inr, 2),
            "verdict": _verdict(float(p)),
        }
        for i, p in enumerate(preds)
    ]
    results.sort(key=lambda r: r["predicted_roas"], reverse=True)
    return {
        "count": len(results),
        "blended_predicted_roas": round(
            float(sum(r["predicted_revenue_inr"] for r in results)
                  / max(sum(p.planned_budget_inr for p in req.plans), 1)), 3
        ),
        "results": results,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.post("/allocate")
def allocate(req: AllocateRequest) -> dict:
    from adperf.insights import budget as bd

    frame = _to_frame(req.plans)
    candidates = frame.to_dict("records")
    result = bd.compare_to_even_split(STATE["bundle"], candidates, req.total_budget_inr)
    return result


@app.get("/insights")
def insights(dimension: str | None = None) -> dict:
    pack = STATE.get("insights")
    if pack is None:
        raise HTTPException(404, "insights not generated - run adperf.insights.marketing")
    if dimension:
        if dimension not in pack["dimension_uplift"]:
            raise HTTPException(404, f"unknown dimension: {dimension}")
        return {"dimension": dimension, "uplift": pack["dimension_uplift"][dimension]}
    return {
        "generated_at": pack["generated_at"],
        "recommendations": pack["recommendations"],
        "available_dimensions": list(pack["dimension_uplift"].keys()),
    }


@app.get("/monitoring/drift")
def drift() -> dict:
    path = C.ARTIFACT_DIR / "drift_report.json"
    if not path.exists():
        raise HTTPException(404, "no drift report - run adperf.monitoring.drift")
    report = json.loads(path.read_text())
    return {
        "verdict": report["verdict"],
        "performance": report["performance"],
        "target_drift": report["target_drift"],
        "alerting_features": [f for f in report["feature_drift"] if f["status"] == "ALERT"],
    }
