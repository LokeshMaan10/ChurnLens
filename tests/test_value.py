"""Phase 8: CLTV formulas, price fallbacks, segment rates, Cox covariates."""

import numpy as np
import pandas as pd
import pytest

from churnlens.value import cltv, cox


def test_monthly_discount_compounds_to_annual() -> None:
    d = cltv.monthly_discount(0.10)
    assert (1 + d) ** 12 == pytest.approx(1.10)


def test_cltv_formula_closed_form() -> None:
    # r = 1 (never churns), no discount: CLTV = m * H
    s = cltv.formula_survival(np.array([1.0]), 12)
    assert cltv.cltv_from_survival(np.array([100.0]), s, 0.0)[0] == pytest.approx(1200)
    # r = 0.5, H = 3, d = 0: m * (1 + 0.5 + 0.25)
    s = cltv.formula_survival(np.array([0.5]), 3)
    assert cltv.cltv_from_survival(np.array([100.0]), s, 0.0)[0] == pytest.approx(175)


def test_monthly_price_fallbacks() -> None:
    f = pd.DataFrame({"last_plan_days": [30, 30, 7, np.nan],
                      "last_amount_paid": [149, 0, 0, np.nan],
                      "last_list_price": [149, 99, 0, np.nan]})
    price = cltv.monthly_price(f, median_price=149.0, days_per_month=30)
    assert price.tolist() == pytest.approx([149, 99, 149, 149])


def test_small_segments_use_overall_rate() -> None:
    seg = pd.Series(["a"] * 300 + ["b"] * 5)
    churn = np.array([0] * 270 + [1] * 30 + [1] * 5)
    r = cltv.segment_renewal_rates(seg, churn, min_users=200).set_index("segment")
    assert r.loc["a", "renewal_rate"] == pytest.approx(0.9)
    assert r.loc["b", "uses_overall_rate"]
    assert r.loc["b", "renewal_rate"] == pytest.approx(1 - 35 / 305)


def test_backtest_metrics() -> None:
    out = cltv.backtest(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
    assert out["mae"] == pytest.approx(10) and out["bias"] == pytest.approx(0)
    assert out["total_ratio"] == pytest.approx(1.0)


def test_cox_covariates() -> None:
    spells = pd.DataFrame({"first_auto_renew": [1, 0], "first_paid": [149, 0],
                           "first_list_price": [149, 99], "first_plan_days": [30, 7],
                           "channel": ["channel 7", "channel 3"]})
    X = cox.covariates(spells, ["channel 7"])
    assert X.loc[0, "auto_renew"] == 1 and X.loc[1, "free_start"] == 1
    assert X.loc[1, "plan_under_30d"] == 1 and X.loc[0, "price_per_day"] == pytest.approx(149 / 30)
    assert list(X["channel_7"]) == [1, 0]
