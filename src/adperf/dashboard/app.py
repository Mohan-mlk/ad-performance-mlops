"""Streamlit dashboard for the ASR media team.

Five tabs, matching the media workflow: plan a campaign, split the budget, read
insights, check whether the model is trustworthy, and review its MLflow lineage.
The monitoring tab is
deliberately in front of the business, not hidden in a Grafana no one opens --
the people spending the money should be able to see when the model is drifting.

Run:  streamlit run src/adperf/dashboard/app.py
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from adperf import config as C
from adperf.insights import budget as bd

st.set_page_config(page_title="ASR Ad Performance", page_icon="💍", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(C.MODEL_PATH)


@st.cache_data
def load_weekly():
    df = pd.read_parquet(C.DATA_DIR / "processed" / "weekly.parquet")
    for col in C.CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df


@st.cache_data
def load_json(path: str):
    p = C.ARTIFACT_DIR / path
    return json.loads(p.read_text()) if p.exists() else None



@st.cache_data(ttl=20)
def load_mlflow_experiments():
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=C.MLFLOW_TRACKING_URI)
    experiments = client.search_experiments()
    return [
        {"experiment_id": exp.experiment_id, "name": exp.name}
        for exp in experiments
        if exp.lifecycle_stage == "active"
    ]


@st.cache_data(ttl=20)
def load_mlflow_runs(experiment_id: str):
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=C.MLFLOW_TRACKING_URI)
    runs = client.search_runs(
        [experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )
    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", "Unnamed run"),
            "status": run.info.status,
            "started_at": run.info.start_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }
        for run in runs
    ]


def predict_one(bundle, plan: dict) -> float:
    frame = pd.DataFrame([plan])
    X = frame[bundle["features"]].copy()
    for col, cats in bundle["categories"].items():
        X[col] = pd.Categorical(X[col], categories=cats)
    return float(bundle["model"].predict(X)[0])


def base_plan(weekly: pd.DataFrame) -> dict:
    """Median ad-set week, used to fill everything the user does not set."""
    plan = {}
    for col in C.NUMERIC_FEATURES:
        plan[col] = float(weekly[col].median())
    for col in C.CATEGORICAL_FEATURES:
        plan[col] = weekly[col].mode().iloc[0]
    return plan


bundle = load_bundle()
weekly = load_weekly()
insights = load_json("insights.json")
drift = load_json("drift_report.json")
backtest = load_json("backtest_summary.json")

st.title("ASR Jewellery Works — Ad Performance & Marketing Insights")
st.info(
    "**Demo data only:** these forecasts use generated campaign records, not live "
    "advertising-account data. They demonstrate the planning workflow and are not "
    "a guarantee of future ROAS."
)
trained_at = pd.Timestamp(bundle["trained_at"]).tz_convert("UTC")
st.caption(
    f"Model trained {trained_at:%Y-%m-%d %H:%M UTC} · "
    f"{bundle['train_rows']:,} weekly training records · "
    f"{len(bundle['features'])} pre-launch features"
)
m = bundle["metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model MAE (ROAS)", f"{m.get('test_mae', 0):.2f}")
c2.metric("Beats category baseline by", f"{m.get('mae_improvement_vs_category_pct', 0):.1f}%")
c3.metric("Top-20% pick lift", f"{m.get('test_top20_lift', 0):.2f}x")
if drift:
    c4.metric("Monitoring", drift["verdict"]["action"])

tab_plan, tab_budget, tab_insight, tab_monitor, tab_mlflow = st.tabs(
    ["Campaign planner", "Budget allocator", "Marketing insights", "Model monitoring", "MLflow lineage"]
)

# ---------------------------------------------------------------- planner
with tab_plan:
    st.subheader("What ROAS should I expect if I run this?")
    plan = base_plan(weekly)
    a, b, c = st.columns(3)
    with a:
        plan["product_category"] = st.selectbox("Product category", C.PRODUCT_CATEGORIES, index=1)
        plan["geo"] = st.selectbox("Geo", C.GEOS)
        plan["audience_segment"] = st.selectbox("Audience segment", C.AUDIENCE_SEGMENTS)
    with b:
        plan["creative_language"] = st.selectbox("Creative language", C.CREATIVE_LANGUAGES)
        plan["creative_format"] = st.selectbox("Creative format", C.CREATIVE_FORMATS, index=2)
        plan["platform"] = st.selectbox("Platform", C.PLATFORMS)
    with c:
        plan["age_bracket"] = st.selectbox("Age bracket", C.AGE_BRACKETS, index=1)
        plan["bid_strategy"] = st.selectbox("Bid strategy", C.BID_STRATEGIES, index=2)
        plan["festival_window"] = st.selectbox("Festival window", ["none"] + list(C.FESTIVALS))

    d, e = st.columns(2)
    plan["planned_budget_inr"] = d.number_input("Weekly budget (INR)", 1_000, 500_000, 25_000, 1_000)
    plan["days_to_festival"] = e.slider("Days to festival", 0, 60, 21)

    roas = predict_one(bundle, plan)
    revenue = roas * plan["planned_budget_inr"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted ROAS", f"{roas:.2f}x")
    k2.metric("Predicted revenue", f"₹{revenue:,.0f}")
    k3.metric("Verdict", "Scale" if roas >= 8 else ("Run" if roas >= C.HIGH_PERFORMER_THRESHOLD else "Hold"))

    st.caption(
        "Predictions are weekly and pre-launch: no click or conversion data is used, "
        "so this is a plan-time forecast rather than an in-flight readout."
    )

    st.markdown("**Swap one setting at a time** — the counterfactual behind the recommendation:")
    dim = st.selectbox("Vary", [d for d in C.CATEGORICAL_FEATURES if d != "festival_window"])
    rows = []
    for level in bundle["categories"][dim]:
        alt = {**plan, dim: level}
        rows.append({dim: level, "predicted_roas": round(predict_one(bundle, alt), 3)})
    comp = pd.DataFrame(rows).sort_values("predicted_roas", ascending=False)
    st.bar_chart(comp.set_index(dim))

    st.markdown("**Budget response curve** — where this ad set saturates:")
    curve = pd.DataFrame({"budget_inr": [5_000 * i for i in range(1, 21)]})
    curve["predicted_roas"] = [
        predict_one(bundle, {**plan, "planned_budget_inr": b}) for b in curve["budget_inr"]
    ]
    curve["predicted_revenue_inr"] = curve["predicted_roas"] * curve["budget_inr"]
    st.line_chart(curve.set_index("budget_inr")[["predicted_roas"]])

# ---------------------------------------------------------------- budget
with tab_budget:
    st.subheader("Split a weekly budget across candidate ad sets")
    total = st.number_input("Total weekly budget (INR)", 10_000, 5_000_000, 300_000, 10_000)
    cats = st.multiselect("Categories to consider", C.PRODUCT_CATEGORIES,
                          default=["bridal_set", "gold_coin", "gold_chain", "daily_wear_gold"])
    geos = st.multiselect("Geos to consider", C.GEOS, default=["vijayawada", "guntur", "hyderabad"])

    if st.button("Optimise allocation") and cats and geos:
        base = base_plan(weekly)
        candidates = [
            {**base, "product_category": cat, "geo": geo,
             "creative_language": "telugu" if geo in ("vijayawada", "guntur") else "telugu_english_mix"}
            for cat in cats
            for geo in geos
        ]
        with st.spinner(f"Optimising across {len(candidates)} candidates..."):
            result = bd.compare_to_even_split(bundle, candidates, float(total))
        x, y = st.columns(2)
        x.metric("Predicted revenue (optimised)", f"₹{result['optimised_revenue_inr']:,.0f}")
        y.metric("Vs even split", f"{result['uplift_pct']:+.1f}%")
        alloc = pd.DataFrame(result["allocation"])
        st.dataframe(
            alloc[alloc["allocated_budget_inr"] > 0][
                ["product_category", "geo", "creative_language", "allocated_budget_inr",
                 "predicted_roas", "predicted_revenue_inr", "budget_share_pct", "at_budget_cap"]
            ],
            width="stretch",
        )
        capped = int(alloc["at_budget_cap"].sum())
        st.caption(
            "Solved by dynamic programming over a discretised budget, so it makes no assumption "
            "that revenue curves are concave. Per-ad-set spend is capped at the largest weekly "
            "budget the model was trained on — beyond that a tree model just repeats its last "
            "answer, and the optimiser would happily put everything into one audience."
        )
        if capped:
            st.info(
                f"{capped} ad set(s) hit the budget cap — the optimiser wants to spend more there "
                "than the model can vouch for. Scale those up gradually and let the next retrain "
                "learn the higher range."
            )

# ---------------------------------------------------------------- insights
with tab_insight:
    if not insights:
        st.warning("Run `python -m adperf.insights.marketing` to generate the insight pack.")
    else:
        st.subheader("Recommendations")
        for r in insights["recommendations"]:
            st.markdown(f"- {r}")

        st.subheader("Creative language by geo")
        lg = pd.DataFrame(insights["language_by_geo"])
        st.dataframe(
            lg.pivot(index="creative_language", columns="geo", values="predicted_roas").round(2),
            width="stretch",
        )

        st.subheader("Festival timing")
        timing = pd.DataFrame(insights["festival_timing"])
        st.line_chart(timing.set_index("days_to_festival"))

        st.subheader("Creative fatigue")
        st.dataframe(pd.DataFrame(insights["creative_fatigue"]).round(3), width="stretch")

        st.subheader("Best performing segments (observed)")
        st.dataframe(pd.DataFrame(insights["segment_leaderboard"]).round(2), width="stretch")

# ---------------------------------------------------------------- monitoring
with tab_monitor:
    if not drift:
        st.warning("Run `python -m adperf.monitoring.drift` to generate the drift report.")
    else:
        v = drift["verdict"]
        colour = {"HEALTHY": st.success, "MONITOR": st.info,
                  "RETRAIN_SOON": st.warning, "RETRAIN_NOW": st.error,
                  "INVESTIGATE": st.error}.get(v["action"], st.info)
        colour(f"**{v['action']}** — {v['reason']}")

        p = drift["performance"]
        x, y, z = st.columns(3)
        x.metric("Baseline MAE", f"{p['baseline_mae']:.2f}")
        y.metric("Live MAE", f"{p['live_mae']:.2f}", f"{p['mae_degradation_pct']:+.1f}%")
        z.metric("Mean ROAS shift", f"{drift['target_drift']['mean_shift_pct']:+.1f}%")

        st.subheader("Feature drift")
        st.caption(
            "Calendar columns are marked SEASONAL and never escalate: comparing any two "
            "time windows moves them by construction, so alerting on them would be noise."
        )
        st.dataframe(pd.DataFrame(drift["feature_drift"]), width="stretch")

        if backtest:
            st.subheader("Rolling-origin backtest")
            st.caption("Mean ± std across walk-forward folds — the honest read on model quality.")
            st.json(backtest)

# ---------------------------------------------------------------- MLflow
with tab_mlflow:
    st.subheader("Training runs from the connected MLflow tracker")
    st.caption(f"Tracking URI: `{C.MLFLOW_TRACKING_URI}` · refreshes every 20 seconds")
    try:
        experiments = load_mlflow_experiments()
    except Exception as exc:  # noqa: BLE001 - show tracker status in the dashboard
        st.error(f"Cannot reach the MLflow tracking server: {exc}")
        st.info("Start the MLflow service with `docker compose up -d mlflow` and refresh this page.")
    else:
        if not experiments:
            st.info("No active MLflow experiments are available yet. Run a training job to create one.")
        else:
            names = {exp["name"]: exp["experiment_id"] for exp in experiments}
            exp_name = st.selectbox("Experiment", list(names))
            experiment_id = names[exp_name]
            try:
                runs = load_mlflow_runs(experiment_id)
            except Exception as exc:  # noqa: BLE001 - keep the rest of the dashboard usable
                st.error(f"Could not load MLflow runs: {exc}")
                runs = []

            if not runs:
                st.info("This experiment has no training runs yet.")
            else:
                run_rows = []
                for run in runs:
                    started = run["started_at"]
                    run_rows.append({
                        "Run": run["run_name"],
                        "Status": run["status"],
                        "Started (UTC)": (
                            pd.to_datetime(started, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M")
                            if started else "—"
                        ),
                        "Test MAE": run["metrics"].get("test_mae"),
                        "Test R²": run["metrics"].get("test_r2"),
                        "Top 20% lift": run["metrics"].get("test_top20_lift"),
                        "Run ID": run["run_id"],
                    })
                st.dataframe(pd.DataFrame(run_rows), width="stretch", hide_index=True)

                run_by_id = {run["run_id"]: run for run in runs}
                selected_run_id = st.selectbox(
                    "Run details",
                    list(run_by_id),
                    format_func=lambda run_id: run_by_id[run_id]["run_name"],
                )
                selected_run = run_by_id[selected_run_id]
                left, right = st.columns(2)
                left.markdown("**Parameters**")
                left.json(selected_run["params"])
                right.markdown("**Metrics**")
                right.json(selected_run["metrics"])

                from mlflow.tracking import MlflowClient

                client = MlflowClient(tracking_uri=C.MLFLOW_TRACKING_URI)
                try:
                    artifacts = client.list_artifacts(selected_run_id)
                except Exception as exc:  # noqa: BLE001 - artifact listing is supplementary
                    st.warning(f"Run details loaded, but its artifact list is unavailable: {exc}")
                else:
                    st.markdown("**Logged artifacts**")
                    st.dataframe(
                        pd.DataFrame([
                            {"Path": item.path, "Directory": item.is_dir, "Size (bytes)": item.file_size}
                            for item in artifacts
                        ]),
                        width="stretch",
                        hide_index=True,
                    )

st.divider()
st.caption("MLOps · predictions use the versioned model artifact · monitoring reports feature drift, target drift, and performance decay separately")
