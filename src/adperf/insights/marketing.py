"""Turn the model into decisions a media buyer can act on.

Feature importance says *which columns the model leans on*. A buyer needs to
know *which setting to pick*, which is a different question. Everything here is
counterfactual: hold a realistic scenario grid fixed, flip one dimension at a
time, and read the change in predicted ROAS. That isolates the effect of the
lever instead of reporting whatever the account happened to spend money on.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from adperf import config as C


def _scenario_grid(df: pd.DataFrame, n: int = 900, seed: int = 7) -> pd.DataFrame:
    """A sample of recent real ad-set weeks, used as the counterfactual base."""
    recent = df.sort_values("week").tail(max(n * 2, 1000))
    return recent.sample(min(n, len(recent)), random_state=seed).reset_index(drop=True)


def _predict(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    X = frame[bundle["features"]].copy()
    for col, cats in bundle["categories"].items():
        X[col] = pd.Categorical(X[col], categories=cats)
    return bundle["model"].predict(X)


def dimension_uplift(bundle: dict, df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Counterfactual: set every scenario to each level of `dimension` and score.

    Returns predicted ROAS per level and its lift against the grid's own mean,
    so the numbers read as "switching to this is worth +X%".
    """
    grid = _scenario_grid(df)
    levels = list(bundle["categories"].get(dimension, sorted(df[dimension].dropna().unique())))
    rows = []
    for level in levels:
        scenario = grid.copy()
        scenario[dimension] = level
        preds = _predict(bundle, scenario)
        rows.append({"level": level, "predicted_roas": float(np.mean(preds))})
    out = pd.DataFrame(rows).sort_values("predicted_roas", ascending=False)
    base = out["predicted_roas"].mean()
    out["lift_vs_average_pct"] = (out["predicted_roas"] / base - 1) * 100
    out["dimension"] = dimension
    return out.reset_index(drop=True)


def interaction_uplift(bundle: dict, df: pd.DataFrame, dim_a: str, dim_b: str) -> pd.DataFrame:
    """Same idea across two dimensions -- e.g. creative language x geo."""
    grid = _scenario_grid(df, n=500)
    levels_a = list(bundle["categories"].get(dim_a, sorted(df[dim_a].dropna().unique())))
    levels_b = list(bundle["categories"].get(dim_b, sorted(df[dim_b].dropna().unique())))
    rows = []
    for a in levels_a:
        for b in levels_b:
            scenario = grid.copy()
            scenario[dim_a], scenario[dim_b] = a, b
            rows.append({dim_a: a, dim_b: b, "predicted_roas": float(_predict(bundle, scenario).mean())})
    return pd.DataFrame(rows)


def festival_timing_curve(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Predicted ROAS against how early the ad set runs before a festival.

    Conditioned on ad sets that are actually inside a festival window. Scoring
    the whole account instead mixes in off-season weeks, where `days_to_festival`
    means nothing, and flattens the curve into useless advice.

    The peak of this curve is the media plan: it says *when* to switch the
    festival creative on, not merely that festivals matter.
    """
    festival_rows = df[df["festival_window"].astype(str) != "none"]
    source = festival_rows if len(festival_rows) >= 200 else df
    grid = _scenario_grid(source, n=500)
    rows = []
    for dtf in range(0, 61, 2):
        scenario = grid.copy()
        scenario["days_to_festival"] = dtf
        rows.append({"days_to_festival": dtf, "predicted_roas": float(_predict(bundle, scenario).mean())})
    return pd.DataFrame(rows)


def segment_leaderboard(df: pd.DataFrame, min_spend: float = 50_000, top_n: int = 12) -> pd.DataFrame:
    """Observed performance by segment, filtered to segments with real spend.

    This is the descriptive counterpart to the counterfactuals: what actually
    happened, with enough spend behind it to be worth reading.
    """
    keys = ["product_category", "creative_language", "creative_format", "geo"]
    g = (
        df.groupby(keys, observed=True)
        .agg(
            spend_inr=("spend_inr", "sum"),
            revenue_inr=("revenue_inr", "sum"),
            leads=("leads", "sum"),
            conversions=("conversions", "sum"),
            ad_weeks=("week", "size"),
        )
        .reset_index()
    )
    g = g[g["spend_inr"] >= min_spend].copy()
    g["roas"] = g["revenue_inr"] / g["spend_inr"]
    g["cost_per_lead_inr"] = g["spend_inr"] / g["leads"].clip(lower=1)
    g["lead_to_sale_pct"] = g["conversions"] / g["leads"].clip(lower=1) * 100
    return g.sort_values("roas", ascending=False).head(top_n).reset_index(drop=True)


def creative_fatigue_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Observed ROAS and CTR by creative age -- tells you when to refresh."""
    d = df.copy()
    d["age_bucket"] = pd.cut(
        d["creative_age_days"],
        bins=[-1, 6, 13, 20, 27, 34, 48, 200],
        labels=["0-6d", "7-13d", "14-20d", "21-27d", "28-34d", "35-48d", "49d+"],
    )
    return (
        d.groupby("age_bucket", observed=True)
        .apply(
            lambda x: pd.Series(
                {
                    "roas": x["revenue_inr"].sum() / max(x["spend_inr"].sum(), 1),
                    "ctr_pct": x["ctr"].mean() * 100,
                    "spend_inr": x["spend_inr"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def _recommendations(
    uplifts: dict[str, pd.DataFrame], timing: pd.DataFrame, lang_geo: pd.DataFrame
) -> list[str]:
    """Plain-language actions, generated from the counterfactual tables."""
    recs = []
    for dim, tbl in uplifts.items():
        best, worst = tbl.iloc[0], tbl.iloc[-1]
        gap = best["predicted_roas"] / max(worst["predicted_roas"], 0.01) - 1
        if gap < 0.08:  # dimension barely matters; do not waste the buyer's attention
            continue
        recs.append(
            f"{dim.replace('_', ' ').title()}: shift budget to '{best['level']}' "
            f"({best['predicted_roas']:.2f}x predicted, {best['lift_vs_average_pct']:+.0f}% vs average) "
            f"and cut '{worst['level']}' ({worst['predicted_roas']:.2f}x)."
        )

    # Language is the case where the main effect hides the truth: averaged over
    # every geo, Telugu looks flat, because it wins at home and loses in the
    # metro. Report the best language *per geo* instead.
    per_geo = []
    for geo, chunk in lang_geo.groupby("geo", observed=True):
        chunk = chunk.sort_values("predicted_roas", ascending=False)
        best, worst = chunk.iloc[0], chunk.iloc[-1]
        edge = (best["predicted_roas"] / max(worst["predicted_roas"], 0.01) - 1) * 100
        if edge >= 5:
            per_geo.append(f"{geo}: {best['creative_language']} (+{edge:.0f}% over {worst['creative_language']})")
    if per_geo:
        recs.append("Creative language is geo-specific, not global - best per market: " + "; ".join(per_geo) + ".")

    # Report the whole profitable window, not just the peak day: a media plan
    # needs a launch date, and "switch it on 2 days out" is not a plan. If the
    # curve turns out flat, say so rather than dressing noise up as timing advice.
    peak = timing.loc[timing["predicted_roas"].idxmax()]
    floor = timing["predicted_roas"].min()
    spread = (peak["predicted_roas"] / max(floor, 0.01) - 1) * 100
    strong = timing[timing["predicted_roas"] >= 0.95 * peak["predicted_roas"]]
    if spread < 15:
        recs.append(
            f"Festival timing: the curve is flat (only {spread:.0f}% between best and worst lead "
            f"time), so when you launch matters far less than category, geo and creative here."
        )
    else:
        recs.append(
            f"Festival timing: predicted ROAS peaks {int(peak['days_to_festival'])} days out "
            f"({peak['predicted_roas']:.2f}x), {spread:.0f}% above the worst lead time, and holds "
            f"within 5% of peak from {int(strong['days_to_festival'].max())} days out - have "
            f"festival creative live by then and weight budget to the final week."
        )
    return recs


def generate_all(bundle: dict, df: pd.DataFrame) -> dict:
    """Build the full insight pack that the API and dashboard both serve."""
    dims = [
        "product_category",
        "creative_language",
        "creative_format",
        "platform",
        "audience_segment",
        "geo",
        "bid_strategy",
        "age_bracket",
    ]
    uplifts = {d: dimension_uplift(bundle, df, d) for d in dims}
    timing = festival_timing_curve(bundle, df)
    lang_geo = interaction_uplift(bundle, df, "creative_language", "geo")

    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "dimension_uplift": {d: t.to_dict("records") for d, t in uplifts.items()},
        "language_by_geo": lang_geo.to_dict("records"),
        "festival_timing": timing.to_dict("records"),
        "segment_leaderboard": segment_leaderboard(df).to_dict("records"),
        "creative_fatigue": creative_fatigue_curve(df).to_dict("records"),
        "recommendations": _recommendations(uplifts, timing, lang_geo),
    }


def main() -> None:
    import joblib

    bundle = joblib.load(C.MODEL_PATH)
    df = pd.read_parquet(C.DATA_DIR / "processed" / "weekly.parquet")
    for col in C.CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    pack = generate_all(bundle, df)
    C.INSIGHTS_PATH.write_text(json.dumps(pack, indent=2, default=str))
    print(f"wrote {C.INSIGHTS_PATH}")
    for r in pack["recommendations"]:
        print(" -", r)


if __name__ == "__main__":
    main()
