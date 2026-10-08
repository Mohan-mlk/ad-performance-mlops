# Ad Performance Prediction & Marketing Insights — ASR Jewellery Works

An end-to-end MLOps system that forecasts weekly ROAS for planned ad sets, turns
the model into concrete media decisions, splits a budget across campaigns, and
monitors itself well enough to say when it should be retrained.

Built for a jewellery retailer in Vijayawada, so the domain is baked in: eight
product categories with real ticket sizes, Telugu / English / mixed creative,
five Andhra–Telangana markets, and the festival calendar that actually moves
gold sales (Akshaya Tritiya, Dhanteras, wedding season, Sankranti, Ugadi,
Varalakshmi Vratam, Dasara).

---

## Quick start

```bash
pip install -r requirements.txt

make pipeline      # generate data -> features -> train -> insights -> drift
make backtest      # rolling-origin evaluation (the number that counts)
make test          # 23 tests
make api           # FastAPI on :8000, docs at /docs
make dashboard     # Streamlit on :8501
make mlflow-ui     # experiment tracking on :5000

docker compose up  # api + dashboard + mlflow together
```

## Stable public deployment

This repository includes a Render Blueprint that serves the dashboard and API
from the same image that passes the GitHub Actions model-quality gate and API
container smoke check.

1. Push to `main`. After lint, tests, training, rolling-origin backtest, model
   quality gates, and the container smoke check succeed, GitHub Actions pushes
   the image to GitHub Container Registry (GHCR), tagged with the commit SHA
   and `latest`.
2. In GitHub, open the `ad-performance-mlops` package under your profile and
   change its visibility to **Public**. The repository is already public and
   the image contains only this project and its synthetic demo data/model.
3. In Render, create a **Blueprint Instance** from this repository after the
   image package is public. Render reads `render.yaml` and creates the API and
   dashboard services from the GHCR image. Their URLs are listed on each
   service page; the API docs are at the API URL plus `/docs`.
4. Copy each service's **Deploy Hook** from its Render settings. Add GitHub
   Actions repository secrets named `RENDER_API_DEPLOY_HOOK` and
   `RENDER_DASHBOARD_DEPLOY_HOOK`. Later successful `main` builds publish the
   new image and trigger both services to redeploy it.

The Blueprint uses Render's free plan for a student demo. Free services can
sleep while idle; local MLflow databases and prediction logs are not durable
on that plan. Use a persistent managed datastore and paid always-on compute if
you need durable production tracking or logs. Do not upload real customer or
advertising account data to the public repository or image.

---

## What the model actually predicts

**Weekly ROAS for an ad set, at plan time, before a rupee is spent.**

That framing drives three decisions that shape the whole system:

**Pre-launch features only.** Impressions, clicks, CTR and conversions are
outcomes, not inputs. A model that sees them scores beautifully offline and is
useless in the planning meeting, because at planning time none of them exist.
They live in `config.LEAKY_COLUMNS` and `assert_no_leakage()` raises if one ever
reaches the feature list. History enters only through trailing-window encoders
that are shifted a week, so week *t* is scored using weeks strictly before it.

**Weekly grain, not daily.** At ad × day, ~65% of rows bill zero revenue,
because a ₹2.8 lakh bridal set does not sell every day off a ₹4,800 budget.
Rolling up to ad set × week cuts that to 26% and makes ROAS a quantity worth
regressing on.

**Tweedie objective.** Weekly ROAS is zero-inflated but continuous and positive
above zero. Squared error on that shape pulls every prediction toward the middle
and destroys the ranking, which is the entire product. Tweedie (variance power
1.5) is the standard fit — the same objective insurers use for claim cost.

---

## Results

Rolling-origin backtest, six walk-forward folds, retraining at each origin:

| Metric | Result |
| --- | --- |
| MAE (ROAS points) | **5.22 ± 0.24** vs 6.11 baseline |
| Improvement over category-average baseline | **14.6%** |
| Rank correlation (Spearman) | 0.396 ± 0.086 |
| Top-20% pick lift | **1.83x ± 0.20** |
| High-performer AUC | 0.686 ± 0.082 |
| Simulated budget-reallocation uplift | 8.8% ± 2.7% |
| Folds beating the baseline | **6 of 6** |

Reproduce with `make backtest`. Expect small run-to-run movement — LightGBM with
`n_jobs=-1` is not bit-identical across runs — but the ranking conclusions and
the 6-of-6 baseline result have held across every run of this pipeline.

**R² is only ~0.08, and that is the honest headline.** Weekly ad performance is
mostly noise; no model is going to tell you a campaign will return 7.3x rather
than 5.1x. What this one does reliably is *order* ad sets — the top-fifth of its
picks return 1.86x the account average, and it beat the do-nothing baseline in
every fold. That is why the CI gate checks rank correlation and top-quintile
lift, not RMSE.

Single-split metrics are reported too, but **promotion is gated on the
backtest**. One 90-day window swings by more than the differences being judged;
gating on it produces flaky CI and false confidence.

---

## Marketing insights

Feature importance tells you which columns the model leans on. A media buyer
needs to know *which setting to pick*, which is a different question. Every
insight here is counterfactual: hold a grid of real recent ad-set weeks fixed,
flip one dimension, read the change in predicted ROAS. That isolates the lever
instead of reporting whatever the account happened to spend money on.

Current pack (`artifacts/insights.json`, served at `/insights`):

- **Category:** gold coins 10.46x predicted (+63% vs average); diamond rings
  worst at 4.02x.
- **Geo:** Vijayawada 8.20x (+34%); Hyderabad weakest at 4.79x — the metro is
  spillover, not home turf.
- **Creative language is geo-specific, not global.** Telugu wins in every home
  market (+28% in Vijayawada, +27% in Guntur, +21% in Visakhapatnam) but only
  +6% in Hyderabad. Averaged across all geos the effect nearly vanishes, and an
  early version of this report dropped the recommendation entirely. The pack now
  reports best language *per market*.
- **Format:** reels 7.96x (+24%); static images worst at 4.88x.
- **Bid strategy:** target-ROAS bidding +11% over lowest-cost.
- **Festival timing: flat.** Only 7% separates the best and worst lead time, so
  the report says so rather than dressing noise up as a media plan. When you
  launch matters far less than category, geo and creative here.

---

## Budget allocator

Given candidate ad sets and a weekly budget, `/allocate` returns the split that
maximises predicted revenue, with two guardrails.

**Solved by dynamic programming, not greedily.** Handing each increment to
whichever ad set shows the best marginal return is optimal only if every revenue
curve is concave in spend. A gradient-boosted model's curve is a step function,
so greedy gets trapped — it refuses a candidate whose payoff arrives past a
threshold it never reaches one increment at a time. On a real candidate set the
greedy version *lost to an even split by 3.3%*. The DP makes no concavity
assumption and solves the discretised problem exactly.

**Per-ad-set spend is capped at the model's budget support (p95 of observed
weekly spend).** Trees cannot extrapolate: above the largest budget ever seen,
predicted ROAS goes flat, so predicted revenue keeps climbing linearly and the
optimiser will cheerfully put 98% of the budget into a single audience — it did,
claiming +80%. Capped, the same plan funds seven ad sets for a defensible
+43.7%, and ad sets that hit the cap are flagged so they can be scaled up
gradually and picked up by the next retrain.

---

## Monitoring

Three different failures need three different alarms:

| Signal | What it means | Response |
| --- | --- | --- |
| Feature drift (PSI, KS) | the media plan changed | not by itself a reason to retrain |
| Target drift | ROAS moved — gold price ran up, say | recalibrate expectations |
| Performance decay | the only one costing money | needs labels, lags by the conversion window |

`verdict()` encodes the policy so on-call is not guessing: decay **plus** drift
→ `RETRAIN_NOW`; drift alone → `RETRAIN_SOON`; decay with no drift cause →
`INVESTIGATE` (check tracking and attribution before blaming the model).

**Calendar columns never escalate.** The first monitor here alerted on `month`,
`week_of_year` and `days_to_festival` every single run — but all that happened
was the calendar advanced. Drift is measured against a *season-matched* window
one year back, and calendar features are reported as `SEASONAL`. An alarm that
fires every week is an alarm nobody reads.

---

## Layout

```
src/adperf/
  config.py              domain constants, feature lists, leak list, train config
  data/generate.py       causal synthetic account (swap for live ad exports)
  features/build.py      weekly roll-up + leakage-safe trailing encoders
  models/train.py        LightGBM Tweedie + MLflow + baselines + business metric
  models/backtest.py     rolling-origin evaluation
  insights/marketing.py  counterfactual uplift, language x geo, fatigue, timing
  insights/budget.py     DP allocator with extrapolation guardrails
  monitoring/drift.py    PSI / KS / decay + retraining verdict
  serving/api.py         FastAPI: predict, batch, allocate, insights, drift
  dashboard/app.py       Streamlit: planner, allocator, insights, monitoring
tests/                   23 tests
.github/workflows/ci.yml lint -> test -> train -> backtest -> gate -> docker smoke
```

`data/generate.py` stands in for Meta and Google Ads exports until the
connectors are wired up. The process is causal — spend drives impressions,
impressions drive clicks, clicks become showroom enquiries, and only a slice of
those close — so ROAS is derived, never sampled, and a model that learns it is
learning real structure. Replace it with `ingest.py` and nothing downstream
changes as long as the schema holds.

---

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | ROAS, revenue and a run / hold verdict for one plan |
| `POST /predict/batch` | rank up to 500 plans |
| `POST /allocate` | optimal budget split + uplift vs an even split |
| `GET /insights` | recommendations, or one dimension's uplift table |
| `GET /monitoring/drift` | current verdict and alerting features |
| `GET /model-info` | training date, rows, metrics |
| `GET /health` `/ready` | liveness and readiness (readiness fails without a model) |

History features are optional in the request: a brand-new campaign has no
history, so the service fills them from training-set medians rather than
refusing to answer.

---

## Tests worth reading

Several exist because the bug actually happened while building this:

- `test_roas_is_plausible_for_jewellery` — the generator once produced **505x**
  blended ROAS, because a click converted straight into a ₹2.85 lakh sale. The
  enquiry → showroom → close funnel fixed it; this pins blended ROAS under 25x.
- `test_history_encoder_never_sees_its_own_week` — caught a real leak: the
  encoder's smoothing prior was the full-sample mean, so cold-start rows saw the
  future. Subtle, and exactly the kind that inflates offline metrics and then
  fails to reproduce in production. The prior is now trailing-only.
- `test_calendar_features_never_escalate` — pins the seasonality-is-not-drift
  rule described above.
- `test_verdict_requires_cost_not_just_drift` — drift alone must never trigger
  an automatic retrain.
- `test_allocation_respects_the_budget` — the failure that exposed the greedy
  allocator losing to an even split.

---

## Known limitations

- **Synthetic data.** Effects are recovered because they were planted. The
  pipeline, leakage controls and monitoring transfer directly; the coefficients
  do not. Real Meta/Google exports will need attribution-window decisions the
  simulation sidesteps.
- **Last-click revenue.** Jewellery buying is a multi-visit, offline-closing
  journey. Ad-platform revenue over-credits the last touch; a proper MMM or
  geo-holdout would be the honest way to value upper-funnel spend.
- **No in-flight model.** This forecasts at plan time only. A second model that
  updates mid-flight once early CTR and enquiry data arrive is the natural
  follow-up, and would use exactly the columns deliberately excluded here.
- **Festival timing signal is weak.** The counterfactual curve is nearly flat.
  Either the weekly grain washes out the effect or `festival_window` already
  absorbs it — worth digging into on real data before promising timing advice.
- **History encoders still register as drift.** They shift as the account's own
  history evolves, which is expected rather than alarming, and currently
  produces ALERT rows a human has to dismiss.
