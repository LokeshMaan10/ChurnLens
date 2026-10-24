"""Value at risk, risk x value quadrants, rule-based actions, and the budget optimizer."""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pandas as pd

from churnlens.segmentation.rfm import segment_case_sql


def bands(p: np.ndarray, cltv: np.ndarray, risk_threshold: float, value_cut: float
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(risk_band, value_band, quadrant label) per user."""
    risk = np.where(p >= risk_threshold, "high", "low")
    value = np.where(cltv >= value_cut, "high", "low")
    quadrant = np.char.add(np.char.add(risk.astype(str), " risk / "),
                           np.char.add(value.astype(str), " value"))
    return risk, value, quadrant


def validate_rules(rules: list[dict[str, str]], actions: dict[str, Any]) -> None:
    """Every rule names a known action and the list ends with a catch-all."""
    unknown = [r["action"] for r in rules if r["action"] not in actions]
    if unknown:
        raise ValueError(f"rules reference unknown actions: {unknown}")
    if rules[-1]["when"].strip().lower() != "true":
        raise ValueError("action rules must end with a catch-all rule (when: true)")


def assign_actions(df: pd.DataFrame, rules: list[dict[str, str]],
                   actions: dict[str, Any]) -> pd.DataFrame:
    """Apply rules in order (first match wins). Returns rule name and action per row."""
    validate_rules(rules, actions)
    rule_case = segment_case_sql([{"name": r["name"], "when": r["when"]} for r in rules])
    act_case = segment_case_sql([{"name": r["action"], "when": r["when"]} for r in rules])
    con = duckdb.connect()
    con.register("users", df)
    out = con.execute(f"SELECT {rule_case} AS rule, {act_case} AS action FROM users").df()
    con.close()
    return out.set_axis(df.index)


def expected_benefit(evar: np.ndarray, action: pd.Series, actions: dict[str, Any]
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(cost, expected benefit = EVaR x save_rate - cost) for each user's action."""
    cost = action.map({k: v["cost"] for k, v in actions.items()}).to_numpy(dtype=float)
    save = action.map({k: v["save_rate"] for k, v in actions.items()}).to_numpy(dtype=float)
    return cost, evar * save - cost


def _fill(order: np.ndarray, cost: np.ndarray, budget: float, capacity: int) -> np.ndarray:
    """Take items in `order` while the cap allows, skipping any that exceed the budget."""
    chosen = np.zeros(len(cost), dtype=bool)
    spent, count = 0.0, 0
    for i in order:
        if count >= capacity:
            break
        if spent + cost[i] <= budget:
            chosen[i] = True
            spent += cost[i]
            count += 1
    return chosen


def greedy_select(cost: np.ndarray, benefit: np.ndarray, budget: float, capacity: int,
                  lambdas: np.ndarray | None = None) -> np.ndarray:
    """Pick users to maximise total expected benefit under a budget AND a contact cap.

    Candidates have positive benefit and positive cost. For each lambda (shadow price of
    budget) users are ranked by benefit - lambda x cost and filled greedily; the selection
    with the highest total benefit wins. lambda = 0 is optimal-in-spirit when the cap
    binds (take the largest benefits), large lambda approaches benefit-per-NTD ranking
    when the budget binds. Ties break by position, so results are deterministic.
    """
    cand = np.flatnonzero((benefit > 0) & (cost > 0))
    if lambdas is None:
        lambdas = np.concatenate([[0.0], np.logspace(-3, 3, 60)])
    best, best_total = np.zeros(len(cost), dtype=bool), -np.inf
    for lam in lambdas:
        score = benefit[cand] - lam * cost[cand]
        order = cand[np.lexsort((cand, -score))]
        chosen = _fill(order, cost, budget, capacity)
        total = float(benefit[chosen].sum())
        if total > best_total + 1e-9:
            best, best_total = chosen, total
    return best


def lp_upper_bound(cost: np.ndarray, benefit: np.ndarray, budget: float, capacity: int) -> float:
    """Upper bound on achievable benefit: the better (smaller) of the fractional-knapsack
    optimum under the budget alone and the best `capacity` items under the cap alone."""
    cand = (benefit > 0) & (cost > 0)
    b, c = benefit[cand], cost[cand]
    order = np.argsort(-(b / c), kind="stable")
    spent, total = 0.0, 0.0
    for i in order:
        if spent + c[i] <= budget:
            spent += c[i]
            total += b[i]
        else:
            total += b[i] * (budget - spent) / c[i]
            break
    cap_bound = float(np.sort(b)[::-1][:capacity].sum())
    return float(min(total, cap_bound))


def baseline_select(score: np.ndarray, cost: np.ndarray, benefit: np.ndarray, budget: float,
                    capacity: int) -> np.ndarray:
    """Comparison policy: contact in descending `score` (e.g. P(churn) or EVaR) among
    users with a paid action, within the same budget and cap (benefit not consulted)."""
    cand = np.flatnonzero(cost > 0)
    return _fill(cand[np.argsort(-score[cand], kind="stable")], cost, budget, capacity)
