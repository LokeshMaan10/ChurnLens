"""Phase 6: feature selection, calibration, cost threshold, evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnlens.config import load_config
from churnlens.features.registry import all_specs
from churnlens.modeling import data, evaluate, threshold, train

COSTS = {"offer_cost": 30, "churn_loss": 600, "save_rate": 0.25}


def test_model_columns_exclude_demographics_ids_and_labels() -> None:
    specs = all_specs(load_config("features"))
    num, cat = data.model_columns(specs)
    cols = set(num) | set(cat)
    assert not cols & {"age", "gender", "city", "msno", "is_churn"}
    num_d, cat_d = data.model_columns(specs, include_demographics=True)
    assert {"age", "gender", "city"} <= set(num_d) | set(cat_d)


def test_realized_cost_and_breakeven() -> None:
    y = np.array([1, 1, 0, 0])
    contact = np.array([True, False, True, False])
    # contacted: 2 x 30 offer + 1 churner x 0.75 x 600; not contacted: 1 churner x 600
    assert threshold.realized_cost(y, contact, COSTS) == pytest.approx(60 + 450 + 600)
    assert threshold.breakeven(COSTS) == pytest.approx(0.2)


def test_threshold_lands_near_breakeven_for_calibrated_probabilities() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)   # perfectly calibrated by construction
    t = threshold.choose_threshold(y, p, COSTS, step=0.005)
    assert abs(t - threshold.breakeven(COSTS)) <= 0.02


def test_policy_summary_baselines() -> None:
    y = np.array([1, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.1, 0.1])
    s = threshold.policy_summary(y, p, 0.5, COSTS).set_index("policy")
    assert s.loc["contact nobody", "total_cost"] == 600
    assert s.loc["contact everybody", "total_cost"] == 4 * 30 + 0.75 * 600
    assert s.loc["model (contact if p >= threshold)", "total_cost"] == 30 + 0.75 * 600


def test_calibrators_fix_inflated_probabilities() -> None:
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0, 0.3, 50_000)
    y = (rng.uniform(0, 1, true_p.size) < true_p).astype(int)
    inflated = np.sqrt(true_p)          # monotone but badly calibrated
    for method in ("isotonic", "sigmoid"):
        cal = train.cross_calibrate(inflated, y, method, n_folds=5, seed=0)
        assert abs(cal.mean() - y.mean()) < 0.005
        assert np.mean((cal - y) ** 2) < np.mean((inflated - y) ** 2)


def test_calibration_ties_go_to_the_preferred_method() -> None:
    pref = ["sigmoid", "isotonic"]
    assert train.pick_method({"isotonic": 0.03970, "sigmoid": 0.03975}, pref, 0.0001) \
        == "sigmoid"                                   # within tolerance -> preference
    assert train.pick_method({"isotonic": 0.0390, "sigmoid": 0.0400}, pref, 0.0001) \
        == "isotonic"                                  # clearly better -> wins


def test_ranking_metrics_ignore_ties_created_by_calibration() -> None:
    rng = np.random.default_rng(2)
    score = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, score.size) < score ** 3).astype(int)
    stepped = np.round(score, 1)                       # a calibrator that creates ties
    m = evaluate.metrics(y, score, stepped)
    assert m["pr_auc"] == pytest.approx(evaluate.metrics(y, score, score)["pr_auc"])
    assert m["pr_auc"] > evaluate.metrics(y, stepped, stepped)["pr_auc"]
    assert m["mean_predicted"] == pytest.approx(stepped.mean())


def test_top_k_and_lift() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)       # the two churners are ranked first
    tk = evaluate.top_k(y, p, [10, 20, 50]).set_index("top_pct")
    assert tk.loc[20, "precision"] == 1.0 and tk.loc[20, "recall"] == 1.0
    assert tk.loc[20, "lift"] == pytest.approx(5.0)
    lift = evaluate.lift_table(y, p, p, bins=5)
    assert lift["cumulative_gain"].iloc[0] == 1.0 and lift["cumulative_gain"].iloc[-1] == 1.0
    assert lift["users"].sum() == 10


def test_confusion_and_fairness_rates() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.9, 0.1])
    c = evaluate.confusion(y, p, 0.5)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)
    f = evaluate.fairness_table(pd.Series(["a", "a", "b", "b"]), y, p, 0.5, min_n=200)
    a = f.set_index("group").loc["a"]
    assert a["recall"] == 0.5 and a["fnr"] == 0.5 and a["small_group"]


def test_param_distributions_parse() -> None:
    d = data.param_distributions({"a": {"log_uniform": [0.01, 1.0]}, "b": {"choice": [1, 2]}})
    assert d["b"] == [1, 2] and 0.01 <= d["a"].rvs(random_state=0) <= 1.0
    with pytest.raises(ValueError):
        data.param_distributions({"c": {"normal": [0, 1]}})
