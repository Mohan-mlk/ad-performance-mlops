"""Generate a realistic ad-event dataset for ASR Jewellery Works.

This stands in for the real Meta/Google Ads exports until the API connectors are
wired up. The generative process is deliberately *causal*: spend and audience
settings drive impressions, impressions drive clicks, clicks drive conversions,
and revenue is conversions x category AOV. ROAS is therefore a derived quantity,
never sampled directly, so a model that learns it is learning real structure.

Swap this module for `ingest.py` when live exports land; nothing downstream
changes as long as the schema in `schema.py` holds.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from adperf import config as C


def _festival_calendar(start: date, end: date) -> pd.DataFrame:
    """Expand the festival table into concrete dates across the horizon."""
    rows = []
    for year in range(start.year - 1, end.year + 2):
        for name, mmdd in C.FESTIVALS.items():
            month, day = (int(x) for x in mmdd.split("-"))
            # Move each festival by a few days per year: lunar calendars drift.
            jitter = int(np.sin(year * 7 + hash(name) % 11) * 9)
            try:
                d = date(year, month, day) + timedelta(days=jitter)
            except ValueError:
                continue
            rows.append({"festival": name, "festival_date": d})
    return pd.DataFrame(rows).sort_values("festival_date").reset_index(drop=True)


def _nearest_festival(day: date, cal: pd.DataFrame) -> tuple[str, int]:
    """Nearest upcoming festival and days until it (capped at 60 = 'off-season')."""
    future = cal[cal["festival_date"] >= day]
    if future.empty:
        return "none", 60
    row = future.iloc[0]
    delta = (row["festival_date"] - day).days
    if delta > 60:
        return "none", 60
    return row["festival"], int(delta)


def _gold_price_series(dates: list[date], rng: np.random.Generator) -> pd.DataFrame:
    """Gold price per gram (22K, INR) as a drifting random walk with momentum."""
    n = len(dates)
    shocks = rng.normal(0, 38, n) + 2.4  # mild upward drift, as in 2023-2026
    price = 5_600 + np.cumsum(shocks)
    s = pd.Series(price, index=pd.to_datetime(dates))
    momentum = (s / s.rolling(30, min_periods=1).mean() - 1.0) * 100
    return pd.DataFrame(
        {"date": s.index, "gold_price_per_gram": s.values, "gold_price_momentum_30d": momentum.values}
    )


# Multiplicative effects. Each dict is "lift vs. the account average".
_PLATFORM_CPM = {
    "meta_instagram": 182.0,
    "meta_facebook": 148.0,
    "google_search": 310.0,
    "google_pmax": 205.0,
    "youtube": 122.0,
}
_PLATFORM_INTENT = {
    "meta_instagram": 1.00,
    "meta_facebook": 0.92,
    "google_search": 1.62,  # high intent, high cost
    "google_pmax": 1.18,
    "youtube": 0.83,
}
_FORMAT_CTR = {
    "static_image": 0.85,
    "carousel": 1.05,
    "reel": 1.42,
    "short_video": 1.30,
    "long_video": 0.78,
}
_FORMAT_CVR = {
    "static_image": 0.92,
    "carousel": 1.12,
    "reel": 1.08,
    "short_video": 1.15,
    "long_video": 1.25,  # fewer clicks, but they convert (bridal storytelling)
}
_GEO_AFFINITY = {  # ASR's home turf converts better than metro spillover
    "vijayawada": 1.38,
    "guntur": 1.22,
    "visakhapatnam": 1.05,
    "hyderabad": 0.88,
    "rest_of_ap_ts": 0.79,
}
_LANG_GEO_LIFT = {
    ("telugu", "vijayawada"): 1.30,
    ("telugu", "guntur"): 1.27,
    ("telugu", "visakhapatnam"): 1.18,
    ("telugu", "hyderabad"): 0.96,
    ("telugu", "rest_of_ap_ts"): 1.12,
    ("telugu_english_mix", "hyderabad"): 1.22,
    ("telugu_english_mix", "visakhapatnam"): 1.10,
    ("english", "hyderabad"): 1.14,
    ("english", "vijayawada"): 0.82,
    ("english", "guntur"): 0.78,
}
_SEGMENT_CATEGORY_FIT = {
    ("bridal", "bridal_set"): 1.9,
    ("bridal", "temple_jewellery"): 1.5,
    ("bridal", "bangles_kada"): 1.3,
    ("investment", "gold_coin"): 1.85,
    ("investment", "silver_article"): 1.25,
    ("gifting", "gold_chain"): 1.35,
    ("gifting", "silver_article"): 1.30,
    ("daily_wear", "daily_wear_gold"): 1.6,
    ("daily_wear", "gold_chain"): 1.2,
    ("festival_shopper", "gold_coin"): 1.4,
    ("festival_shopper", "bangles_kada"): 1.3,
}
_AGE_CVR = {"18-24": 0.62, "25-34": 1.24, "35-44": 1.32, "45-54": 1.16, "55+": 0.88}
# Enquiry -> billed sale. Inversely related to ticket size: gold coins close over
# the counter, a bridal set takes three showroom visits and a family decision.
_CATEGORY_CLOSE_RATE = {
    "bridal_set": 0.016,
    "gold_chain": 0.055,
    "temple_jewellery": 0.022,
    "diamond_ring": 0.024,
    "daily_wear_gold": 0.075,
    "bangles_kada": 0.046,
    "gold_coin": 0.130,
    "silver_article": 0.210,
}
_SEGMENT_CLOSE_LIFT = {
    "bridal": 1.18,
    "gifting": 1.05,
    "investment": 1.24,
    "daily_wear": 1.02,
    "festival_shopper": 0.92,
}
_BID_EFFICIENCY = {
    "lowest_cost": 0.94,
    "cost_cap": 1.06,
    "target_roas": 1.19,
    "manual_cpc": 0.88,
}


def generate(
    n_rows: int = 24_000,
    start: str = "2024-01-01",
    end: str = "2026-06-30",
    seed: int = 42,
) -> pd.DataFrame:
    """Return one row per ad x day of delivery."""
    rng = np.random.default_rng(seed)
    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)
    horizon = (end_d - start_d).days
    all_dates = [start_d + timedelta(days=i) for i in range(horizon + 1)]

    cal = _festival_calendar(start_d, end_d)
    gold = _gold_price_series(all_dates, rng).set_index("date")

    # ---- ad sets as persistent entities, each with a flight window ------
    # Real accounts run a few hundred ad sets that deliver for weeks. Modelling
    # them as a panel (ad set x day) is what makes weekly roll-ups, campaign
    # history features and creative fatigue meaningful.
    avg_flight = 26
    n_adsets = max(40, int(n_rows / avg_flight))
    n_campaigns = max(12, n_adsets // 6)
    campaign_ids = [f"ASR-CMP-{i:04d}" for i in range(n_campaigns)]
    campaign_quality = dict(zip(campaign_ids, rng.normal(1.0, 0.16, n_campaigns).clip(0.55, 1.6)))

    ads = pd.DataFrame({"ad_id": [f"ASR-AD-{i:05d}" for i in range(n_adsets)]})
    ads["campaign_id"] = rng.choice(campaign_ids, n_adsets)
    ads["platform"] = rng.choice(C.PLATFORMS, n_adsets, p=[0.28, 0.17, 0.16, 0.22, 0.17])
    ads["product_category"] = rng.choice(
        C.PRODUCT_CATEGORIES, n_adsets, p=[0.15, 0.17, 0.10, 0.09, 0.16, 0.12, 0.13, 0.08]
    )
    ads["creative_language"] = rng.choice(C.CREATIVE_LANGUAGES, n_adsets, p=[0.52, 0.21, 0.27])
    ads["creative_format"] = rng.choice(C.CREATIVE_FORMATS, n_adsets, p=[0.24, 0.18, 0.26, 0.20, 0.12])
    ads["audience_segment"] = rng.choice(C.AUDIENCE_SEGMENTS, n_adsets, p=[0.22, 0.18, 0.17, 0.23, 0.20])
    ads["geo"] = rng.choice(C.GEOS, n_adsets, p=[0.30, 0.18, 0.16, 0.22, 0.14])
    ads["age_bracket"] = rng.choice(C.AGE_BRACKETS, n_adsets, p=[0.14, 0.31, 0.27, 0.18, 0.10])
    ads["gender_target"] = rng.choice(C.GENDER_TARGETS, n_adsets, p=[0.58, 0.14, 0.28])
    ads["bid_strategy"] = rng.choice(C.BID_STRATEGIES, n_adsets, p=[0.34, 0.24, 0.27, 0.15])
    ads["planned_budget_inr"] = np.round(
        np.exp(rng.normal(np.log(4_800), 0.72, n_adsets)).clip(600, 120_000), -1
    )
    ads["audience_size_lakh"] = np.round(
        np.exp(rng.normal(np.log(6.5), 0.55, n_adsets)).clip(0.4, 60), 2
    )
    ads["_launch_offset"] = rng.integers(0, max(1, horizon - 10), n_adsets)
    ads["_flight_days"] = rng.integers(7, 62, n_adsets)

    # Festival campaigns are not launched at random: pull roughly half of the
    # ad sets to start 3-5 weeks ahead of the nearest relevant festival.
    fest_dates = cal["festival_date"].tolist()
    pull = rng.random(n_adsets) < 0.5
    for i in np.where(pull)[0]:
        target = fest_dates[int(rng.integers(0, len(fest_dates)))] - timedelta(
            days=int(rng.integers(21, 36))
        )
        off = (target - start_d).days
        if 0 <= off < horizon - 5:
            ads.loc[i, "_launch_offset"] = off

    ads["_flight_days"] = np.minimum(
        ads["_flight_days"], horizon - ads["_launch_offset"] + 1
    ).clip(lower=1)

    df = ads.loc[ads.index.repeat(ads["_flight_days"])].reset_index(drop=True)
    df["creative_age_days"] = df.groupby("ad_id").cumcount()
    df["date"] = pd.to_datetime(
        [start_d + timedelta(days=int(o + a)) for o, a in zip(df["_launch_offset"], df["creative_age_days"])]
    )
    df = df.drop(columns=["_launch_offset", "_flight_days"])
    n_rows = len(df)

    # Daily budget wobbles around the ad set's planned level.
    df["planned_budget_inr"] = np.round(
        df["planned_budget_inr"].to_numpy() * rng.uniform(0.75, 1.3, n_rows), -1
    )

    # ---- calendar + market context -------------------------------------
    fest = [_nearest_festival(d.date(), cal) for d in df["date"]]
    df["festival_window"] = [f[0] for f in fest]
    df["days_to_festival"] = [f[1] for f in fest]
    df = df.join(gold, on="date")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Salary credit + Sunday shopping is a real pattern for AP retail footfall.
    df["is_payday_window"] = df["date"].dt.day.isin([1, 2, 3, 4, 5, 28, 29, 30, 31]).astype(int)

    # ---- delivery funnel -------------------------------------------------
    cpm = df["platform"].map(_PLATFORM_CPM).to_numpy(dtype=float, copy=True)
    # Auction heats up pre-festival: everyone bids on the same gold buyers.
    cpm = cpm * (1 + 0.25 * np.exp(-df["days_to_festival"].to_numpy() / 12))
    cpm = cpm * rng.lognormal(0, 0.14, n_rows)
    spend = df["planned_budget_inr"].to_numpy() * rng.uniform(0.82, 1.0, n_rows)
    impressions = np.maximum(80, spend / cpm * 1000)

    quality = df["campaign_id"].map(campaign_quality).to_numpy()
    fatigue = np.exp(-df["creative_age_days"].to_numpy() / 55)  # creative wear-out
    lang_geo = np.array(
        [_LANG_GEO_LIFT.get((l, g), 1.0) for l, g in zip(df["creative_language"], df["geo"])]
    )

    ctr = (
        0.0122
        * df["creative_format"].map(_FORMAT_CTR).to_numpy()
        * lang_geo
        * quality
        * (0.75 + 0.5 * fatigue)
        * (1 + 0.22 * np.exp(-df["days_to_festival"].to_numpy() / 15))
        * rng.lognormal(0, 0.24, n_rows)
    ).clip(0.0008, 0.11)
    clicks = rng.binomial(impressions.astype(int).clip(1, 10_000_000), ctr)

    fest_boost = np.array(
        [
            C.FESTIVAL_CATEGORY_BOOST.get(f, {}).get(c, 1.0)
            for f, c in zip(df["festival_window"], df["product_category"])
        ]
    )
    # The boost decays with distance from the festival date.
    fest_boost = 1 + (fest_boost - 1) * np.exp(-df["days_to_festival"].to_numpy() / 18)

    seg_fit = np.array(
        [
            _SEGMENT_CATEGORY_FIT.get((s, c), 0.95)
            for s, c in zip(df["audience_segment"], df["product_category"])
        ]
    )
    # Rising gold price pulls investment buyers in and pushes bridal budgets out.
    mom = df["gold_price_momentum_30d"].to_numpy()
    is_invest = df["product_category"].isin(["gold_coin", "silver_article"]).to_numpy()
    is_bridal = df["product_category"].isin(["bridal_set", "temple_jewellery"]).to_numpy()
    gold_effect = 1 + np.where(is_invest, 0.055, 0.0) * mom - np.where(is_bridal, 0.022, 0.0) * mom

    cvr = (
        0.0092
        * df["creative_format"].map(_FORMAT_CVR).to_numpy()
        * df["geo"].map(_GEO_AFFINITY).to_numpy()
        * df["age_bracket"].map(_AGE_CVR).to_numpy()
        * df["bid_strategy"].map(_BID_EFFICIENCY).to_numpy()
        * df["platform"].map(_PLATFORM_INTENT).to_numpy()
        * seg_fit
        * fest_boost
        * gold_effect.clip(0.6, 1.9)
        * quality
        * (1 + 0.10 * df["is_weekend"].to_numpy())
        * (1 + 0.07 * df["is_payday_window"].to_numpy())
        * rng.lognormal(0, 0.30, n_rows)
    ).clip(0.0004, 0.35)

    # Budget saturation: doubling spend never doubles enquiries.
    saturation = (df["planned_budget_inr"].to_numpy() / 4_800) ** -0.11
    # A click does not buy a 2.8 lakh bridal set. It becomes an enquiry / showroom
    # visit, and only a slice of those close -- lower the higher the ticket.
    leads = rng.binomial(clicks.clip(0, 10_000_000), (cvr * saturation).clip(0.0002, 0.4))
    close = (
        df["product_category"].map(_CATEGORY_CLOSE_RATE).to_numpy(dtype=float, copy=True)
        * df["audience_segment"].map(_SEGMENT_CLOSE_LIFT).to_numpy()
        * (1 + 0.18 * (df["geo"] == "vijayawada").to_numpy())  # walk-in distance to the showroom
        * rng.lognormal(0, 0.22, n_rows)
    ).clip(0.002, 0.55)
    conversions = rng.binomial(leads, close)

    aov = df["product_category"].map(C.CATEGORY_AOV).to_numpy(dtype=float, copy=True) * rng.lognormal(0, 0.21, n_rows)
    # Gold price feeds straight into ticket size.
    aov = aov * (df["gold_price_per_gram"].to_numpy() / 6_000) ** 0.55

    df["impressions"] = impressions.astype(int)
    df["clicks"] = clicks
    df["leads"] = leads
    df["conversions"] = conversions
    df["spend_inr"] = np.round(spend, 2)
    df["revenue_inr"] = np.round(conversions * aov, 2)
    df["ctr"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], 0.0)
    df["cpc_inr"] = np.where(df["clicks"] > 0, df["spend_inr"] / df["clicks"], 0.0)
    df["cpl_inr"] = np.where(df["leads"] > 0, df["spend_inr"] / df["leads"], 0.0)
    df["cpa_inr"] = np.where(df["conversions"] > 0, df["spend_inr"] / df["conversions"], 0.0)
    df["lead_to_sale_rate"] = np.where(df["leads"] > 0, df["conversions"] / df["leads"], 0.0)
    df[C.TARGET] = np.round(df["revenue_inr"] / df["spend_inr"].clip(lower=1.0), 4)
    df["is_high_performer"] = (df[C.TARGET] >= C.HIGH_PERFORMER_THRESHOLD).astype(int)

    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate ASR ad-event data")
    p.add_argument("--rows", type=int, default=24_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(C.RAW_PATH))
    args = p.parse_args()

    df = generate(n_rows=args.rows, seed=args.seed)
    out = pd.Series([args.out]).iloc[0]
    from pathlib import Path

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df):,} rows -> {out}")
    print(f"date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"blended ROAS: {df['revenue_inr'].sum() / df['spend_inr'].sum():.2f}x")


if __name__ == "__main__":
    main()
