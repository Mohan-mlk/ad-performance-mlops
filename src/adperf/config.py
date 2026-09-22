"""Central configuration for the ASR Jewellery Works ad-performance system.

Everything domain-specific lives here so the pipeline code stays generic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("ADPERF_DATA_DIR", ROOT / "data"))
ARTIFACT_DIR = Path(os.getenv("ADPERF_ARTIFACT_DIR", ROOT / "artifacts"))
RAW_PATH = DATA_DIR / "raw" / "ad_events.parquet"
REFERENCE_PATH = DATA_DIR / "reference" / "reference_window.parquet"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
INSIGHTS_PATH = ARTIFACT_DIR / "insights.json"

# MLflow deprecated the bare file store; SQLite is the smallest backend that
# still supports the model registry, and it needs no server to run locally.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "asr-ad-performance")
REGISTERED_MODEL_NAME = "asr_roas_predictor"

# --------------------------------------------------------------------------
# Business domain
# --------------------------------------------------------------------------
PRODUCT_CATEGORIES = [
    "bridal_set",
    "gold_chain",
    "temple_jewellery",
    "diamond_ring",
    "daily_wear_gold",
    "bangles_kada",
    "gold_coin",
    "silver_article",
]

# Average order value in INR, by category. Drives revenue simulation and is a
# genuine business input: ROAS is far more AOV-sensitive than CTR-sensitive.
CATEGORY_AOV = {
    "bridal_set": 285_000,
    "gold_chain": 68_000,
    "temple_jewellery": 125_000,
    "diamond_ring": 92_000,
    "daily_wear_gold": 34_000,
    "bangles_kada": 74_000,
    "gold_coin": 41_000,
    "silver_article": 9_500,
}

CREATIVE_LANGUAGES = ["telugu", "english", "telugu_english_mix"]
CREATIVE_FORMATS = ["static_image", "carousel", "reel", "short_video", "long_video"]
PLATFORMS = ["meta_instagram", "meta_facebook", "google_search", "google_pmax", "youtube"]
AUDIENCE_SEGMENTS = ["bridal", "gifting", "investment", "daily_wear", "festival_shopper"]
GEOS = ["vijayawada", "guntur", "visakhapatnam", "hyderabad", "rest_of_ap_ts"]
AGE_BRACKETS = ["18-24", "25-34", "35-44", "45-54", "55+"]
GENDER_TARGETS = ["female", "male", "all"]
BID_STRATEGIES = ["lowest_cost", "cost_cap", "target_roas", "manual_cpc"]

# Festival / demand windows that actually move jewellery sales in AP & Telangana.
FESTIVALS = {
    "ugadi": "03-30",
    "akshaya_tritiya": "04-30",
    "varalakshmi_vratam": "08-08",
    "dasara": "10-12",
    "dhanteras_diwali": "11-01",
    "wedding_season_peak": "12-05",
    "sankranti": "01-14",
}

# Categories whose demand spikes hardest for each festival window.
FESTIVAL_CATEGORY_BOOST = {
    "akshaya_tritiya": {"gold_coin": 2.1, "daily_wear_gold": 1.5, "gold_chain": 1.4},
    "dhanteras_diwali": {"gold_coin": 1.9, "silver_article": 1.7, "gold_chain": 1.35},
    "wedding_season_peak": {"bridal_set": 2.2, "temple_jewellery": 1.8, "bangles_kada": 1.5},
    "varalakshmi_vratam": {"temple_jewellery": 1.6, "bangles_kada": 1.3},
    "sankranti": {"bangles_kada": 1.4, "daily_wear_gold": 1.3},
    "ugadi": {"daily_wear_gold": 1.25, "gold_chain": 1.2},
    "dasara": {"temple_jewellery": 1.35, "silver_article": 1.2},
}

TARGET = "roas"
HIGH_PERFORMER_THRESHOLD = 3.0  # ROAS at/above this = scale it

CATEGORICAL_FEATURES = [
    "platform",
    "product_category",
    "creative_language",
    "creative_format",
    "audience_segment",
    "geo",
    "age_bracket",
    "gender_target",
    "bid_strategy",
    "festival_window",
]

NUMERIC_FEATURES = [
    "planned_budget_inr",
    "days_to_festival",
    "gold_price_per_gram",
    "gold_price_momentum_30d",
    "creative_age_days",
    "audience_size_lakh",
    "active_days",
    "week_of_year",
    "month",
    "weekend_share",
    "payday_share",
    "hist_roas_category",
    "hist_roas_platform_format",
    "hist_roas_campaign",
    "hist_ctr_creative_lang",
]

# Columns that only exist *after* the money is spent. Never features -- keeping
# this list explicit is what stops a well-meaning teammate from leaking CTR.
LEAKY_COLUMNS = [
    "impressions", "clicks", "leads", "conversions", "spend_inr", "revenue_inr",
    "ctr", "cpc_inr", "cpl_inr", "cpa_inr", "lead_to_sale_rate",
    "roas", "is_high_performer",
]

# Deterministic calendar features. They *always* "drift" when you compare any
# two different time windows, because the calendar moved -- that is seasonality,
# not distribution shift, and alerting on it trains the team to ignore alerts.
# Still reported, never escalated.
HISTORY_WINDOW_WEEKS = int(os.getenv("ADPERF_HISTORY_WINDOW", "12"))

CALENDAR_FEATURES = ["month", "week_of_year", "days_to_festival", "festival_window"]

# Grouping keys for the leakage-safe expanding-mean encoders.
HISTORY_ENCODERS = {
    "hist_roas_category": (["product_category"], "roas"),
    "hist_roas_platform_format": (["platform", "creative_format"], "roas"),
    "hist_roas_campaign": (["campaign_id"], "roas"),
    "hist_ctr_creative_lang": (["creative_language"], "ctr"),
}

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass
class TrainConfig:
    """Hyper-parameters and split policy for the training job."""

    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 40
    subsample: float = 0.9
    colsample_bytree: float = 0.85
    reg_lambda: float = 1.0
    random_state: int = 42
    # Time-ordered split: never evaluate on the past. A 45-day test window left
    # only ~190 rows, which is too few to gate a deploy on -- the fold-to-fold
    # spread was wider than the differences being judged. 90 days roughly
    # triples it; promotion is still decided by the rolling-origin backtest.
    valid_days: int = 60
    test_days: int = 90
    early_stopping_rounds: int = 60
    tags: dict = field(default_factory=lambda: {"business_unit": "ASR Jewellery Works"})
