"""Allocate a weekly budget across candidate ad sets.

Naively you pour everything into the single highest-ROAS ad set. That fails in
practice because returns saturate: the model has `planned_budget_inr` as a
feature, so predicted ROAS falls as budget rises on the same audience. This
allocator exploits that by spending in small increments and always giving the
next increment to whichever ad set currently has the highest *marginal*
predicted revenue -- a greedy solution that is optimal when each ad set's
revenue curve is concave in spend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from adperf import config as C


def _predict(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    X = frame[bundle["features"]].copy()
    for col, cats in bundle["categories"].items():
        X[col] = pd.Categorical(X[col], categories=cats)
    return np.asarray(bundle["model"].predict(X), dtype=float)


def revenue_curve(bundle: dict, candidate: dict, budgets: np.ndarray) -> np.ndarray:
    """Predicted revenue for one ad set across a ladder of weekly budgets."""
    frame = pd.DataFrame([candidate] * len(budgets))
    frame["planned_budget_inr"] = budgets
    return _predict(bundle, frame) * budgets


def allocate(
    bundle: dict,
    candidates: list[dict],
    total_budget: float,
    min_per_adset: float = 0.0,
    max_per_adset: float | None = None,
    step: float | None = None,
    max_grid: int = 120,
    max_share: float = 0.40,
) -> pd.DataFrame:
    """Optimal split of `total_budget` across candidates, by dynamic programming.

    The obvious approach -- hand each increment to whichever ad set has the best
    marginal return -- is only optimal when every revenue curve is concave in
    spend. A gradient-boosted model's curve is a step function, not a smooth
    concave one, so greedy gets trapped: it refuses a candidate whose payoff
    arrives past a threshold it never reaches one increment at a time. On real
    candidate sets that lost to an even split by a few percent.

    This discretises the budget and solves the allocation exactly for that grid
    (a max-plus convolution, the classic resource-allocation DP), which makes no
    concavity assumption. Rows summing to less than the total mean every further
    increment was predicted to lose money -- deliberate underspend is a valid
    answer.
    """
    if not candidates:
        raise ValueError("no candidates supplied")
    n = len(candidates)

    # Two guardrails, both needed, for different reasons.
    #
    # The extrapolation cap is a correctness fix: a tree model predicts a flat
    # ROAS above the largest budget it ever saw, so predicted revenue keeps
    # rising linearly and the optimiser happily puts the entire budget into one
    # ad set. Capping at the 95th percentile of observed weekly spend keeps
    # every prediction inside the range the model can actually speak to.
    #
    # The share cap is a business fix: concentrating a week's media into a
    # single audience courts frequency burn and leaves no read on anything
    # else. Both are overridable by an explicit max_per_adset.
    support = (bundle.get("budget_support") or {}).get("p95")
    if max_per_adset is None:
        limits = [total_budget * max_share]
        if support:
            limits.append(support)
        max_per_adset = min(limits)
    step = step or max(500.0, total_budget / max_grid)
    n_steps = int(total_budget // step)
    if n_steps < n:
        raise ValueError("budget too small relative to the number of candidates")

    cap_steps = int((max_per_adset or total_budget) // step)
    min_steps = int(min_per_adset // step)
    if min_steps * n > n_steps:
        raise ValueError("min_per_adset x candidates exceeds the total budget")

    # Revenue curve per candidate, one batched prediction each.
    ladder = np.arange(0, n_steps + 1) * step
    curves = np.zeros((n, n_steps + 1))
    for i, cand in enumerate(candidates):
        frame = pd.DataFrame([cand] * (n_steps + 1))
        frame["planned_budget_inr"] = np.maximum(ladder, 1.0)
        rev = _predict(bundle, frame) * ladder
        rev[0] = 0.0
        rev[: min_steps] = -np.inf   # below the floor this ad set may not run
        rev[cap_steps + 1 :] = -np.inf
        curves[i] = rev

    NEG = -np.inf
    dp = np.full(n_steps + 1, NEG)
    dp[0] = 0.0
    choice = np.zeros((n, n_steps + 1), dtype=int)

    for i in range(n):
        new_dp = np.full(n_steps + 1, NEG)
        new_choice = np.zeros(n_steps + 1, dtype=int)
        for k in range(min(cap_steps, n_steps) + 1):
            if not np.isfinite(curves[i, k]):
                continue
            cand_vals = dp[: n_steps + 1 - k] + curves[i, k]
            target = new_dp[k:]
            better = cand_vals > target
            target[better] = cand_vals[better]
            new_choice[k:][better] = k
        dp, choice[i] = new_dp, new_choice

    best_b = int(np.nanargmax(np.where(np.isfinite(dp), dp, NEG)))
    alloc_steps = np.zeros(n, dtype=int)
    b = best_b
    for i in range(n - 1, -1, -1):
        k = choice[i, b]
        alloc_steps[i] = k
        b -= k

    rows = []
    for i, cand in enumerate(candidates):
        spend = alloc_steps[i] * step
        rev = curves[i, alloc_steps[i]] if np.isfinite(curves[i, alloc_steps[i]]) else 0.0
        rows.append(
            {
                **{k: cand[k] for k in C.CATEGORICAL_FEATURES if k in cand},
                "allocated_budget_inr": round(float(spend), 2),
                "predicted_roas": round(float(rev / spend) if spend > 0 else 0.0, 3),
                "predicted_revenue_inr": round(float(max(rev, 0.0)), 2),
                "budget_share_pct": round(float(spend) / max(total_budget, 1) * 100, 2),
                "at_budget_cap": bool(spend >= (max_per_adset or total_budget) - step / 2),
            }
        )
    out = pd.DataFrame(rows).sort_values("allocated_budget_inr", ascending=False)
    return out.reset_index(drop=True)


def compare_to_even_split(bundle: dict, candidates: list[dict], total_budget: float) -> dict:
    """Quantify what the optimiser buys over spreading budget evenly."""
    optimised = allocate(bundle, candidates, total_budget)
    even = total_budget / len(candidates)
    even_rev = 0.0
    for cand in candidates:
        frame = pd.DataFrame([cand])
        frame["planned_budget_inr"] = even
        even_rev += float(_predict(bundle, frame)[0] * even)
    opt_rev = optimised["predicted_revenue_inr"].sum()
    return {
        "even_split_revenue_inr": round(even_rev, 2),
        "optimised_revenue_inr": round(opt_rev, 2),
        "uplift_pct": round((opt_rev / max(even_rev, 1) - 1) * 100, 2),
        "allocation": optimised.to_dict("records"),
    }
