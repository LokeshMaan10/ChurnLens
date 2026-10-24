"""Phase 9: bands, rules engine precedence, expected benefit, budget optimizer."""

import numpy as np
import pandas as pd
import pytest

from churnlens.config import load_config
from churnlens.prioritize import engine

ACTIONS = {"monitor": {"cost": 0, "save_rate": 0.0}, "fix": {"cost": 10, "save_rate": 0.2},
           "generic": {"cost": 5, "save_rate": 0.05}}
RULES = [{"name": "low", "when": "risk_band = 'low'", "action": "monitor"},
         {"name": "pay", "when": "driver_feature = 'last_payment_method'", "action": "fix"},
         {"name": "rest", "when": "true", "action": "generic"}]


def test_bands_and_quadrants() -> None:
    risk, value, quad = engine.bands(np.array([0.3, 0.1]), np.array([500, 2000]), 0.25, 1000)
    assert list(risk) == ["high", "low"] and list(value) == ["low", "high"]
    assert list(quad) == ["high risk / low value", "low risk / high value"]


def test_rules_first_match_wins() -> None:
    df = pd.DataFrame({"risk_band": ["low", "high", "high"],
                       "driver_feature": ["last_payment_method", "last_payment_method", None]})
    out = engine.assign_actions(df, RULES, ACTIONS)
    assert list(out["action"]) == ["monitor", "fix", "generic"]   # low risk beats payment rule


def test_rules_validation() -> None:
    with pytest.raises(ValueError, match="catch-all"):
        engine.validate_rules(RULES[:2], ACTIONS)
    with pytest.raises(ValueError, match="unknown actions"):
        engine.validate_rules([{"name": "x", "when": "true", "action": "nope"}], ACTIONS)


def test_configured_rules_are_valid() -> None:
    cfg = load_config("prioritize")
    engine.validate_rules(cfg["rules"], cfg["actions"])
    assert cfg["actions"]["monitor"]["cost"] == 0


def test_expected_benefit() -> None:
    cost, ben = engine.expected_benefit(np.array([1000.0, 1000.0]),
                                        pd.Series(["fix", "monitor"]), ACTIONS)
    assert list(cost) == [10, 0] and ben.tolist() == pytest.approx([190, 0])


def test_optimizer_respects_budget_and_capacity() -> None:
    cost = np.array([10.0, 10.0, 50.0, 5.0, 10.0])
    ben = np.array([100.0, 50.0, 400.0, -1.0, 0.0])
    chosen = engine.greedy_select(cost, ben, budget=30, capacity=10)
    assert list(np.flatnonzero(chosen)) == [0, 1]            # 50-cost item does not fit
    assert cost[chosen].sum() <= 30
    assert engine.lp_upper_bound(cost, ben, 30, 10) >= ben[chosen].sum()


def test_optimizer_uses_contacts_well_when_the_cap_binds() -> None:
    # One slot, ample budget: the 400-benefit user beats the best benefit-per-NTD user.
    cost = np.array([10.0, 10.0, 50.0])
    ben = np.array([100.0, 50.0, 400.0])
    capped = engine.greedy_select(cost, ben, budget=1000, capacity=1)
    assert capped.sum() == 1 and capped[2]
    # Pure ratio ranking (huge lambda only) would waste the slot on the cheap user.
    ratio_only = engine.greedy_select(cost, ben, budget=1000, capacity=1,
                                      lambdas=np.array([1000.0]))
    assert ratio_only[0]


def test_optimizer_prefers_ratio_when_the_budget_binds() -> None:
    cost = np.array([10.0] * 10 + [100.0])
    ben = np.array([60.0] * 10 + [300.0])     # ten cheap items beat one big one on 100 NTD
    chosen = engine.greedy_select(cost, ben, budget=100, capacity=100)
    assert ben[chosen].sum() == 600 and not chosen[-1]
