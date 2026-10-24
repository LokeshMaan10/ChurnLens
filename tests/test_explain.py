"""Phase 7: SHAP mapping and additivity, grouping, reason codes, top drivers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churnlens.config import load_config
from churnlens.explain import groups, reasons
from churnlens.explain.shap_values import column_to_feature, explain

TEMPLATES = load_config("reason_codes")["templates"]


def _toy_model() -> tuple[Pipeline, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "pay": rng.choice(["38", "41", "36"], n)})
    y = ((X["a"] + (X["pay"] == "38") * 1.5 + rng.normal(size=n)) > 1).astype(int)
    prep = ColumnTransformer([("num", "passthrough", ["a", "b"]),
                              ("cat", OneHotEncoder(sparse_output=False), ["pay"])],
                             verbose_feature_names_out=False)
    pipe = Pipeline([("prep", prep), ("clf", LGBMClassifier(n_estimators=30, verbose=-1,
                                                            random_state=0))]).fit(X, y)
    return pipe, X


def test_one_hot_columns_map_back_to_their_feature() -> None:
    cols = ["minutes_7d", "last_payment_method_38", "last_payment_method_infrequent_sklearn",
            "rfm_group_cluster 1"]
    assert column_to_feature(cols, ["minutes_7d"], ["last_payment_method", "rfm_group"]) == \
        ["minutes_7d", "last_payment_method", "last_payment_method", "rfm_group"]
    with pytest.raises(ValueError):
        column_to_feature(["unknown_col"], ["a"], ["b"])


def test_explanation_is_additive_and_feature_level() -> None:
    pipe, X = _toy_model()
    ex = explain(pipe, X, ["a", "b"], ["pay"], log=lambda *_: None)
    assert list(ex.shap.columns) == ["a", "b", "pay"]
    raw = pipe.named_steps["clf"].predict(pipe.named_steps["prep"].transform(X), raw_score=True)
    assert np.allclose(ex.shap.sum(axis=1) + ex.base_value, raw, atol=1e-6)
    # pay=38 raises the outcome in the toy data, so its contribution is positive on average
    assert ex.shap.loc[X["pay"] == "38", "pay"].mean() > 0


def test_correlated_features_share_a_cluster() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=3000)
    X = pd.DataFrame({"m7": base + rng.normal(scale=0.05, size=3000),
                      "m30": base + rng.normal(scale=0.05, size=3000),
                      "other": rng.normal(size=3000), "cat": ["x"] * 3000})
    c = groups.correlation_clusters(X, ["m7", "m30", "other"], ["cat"], 0.8, 3000, 0)
    assert c["m7"] == c["m30"] != c["other"] and c["cat"] == "cat_cat"


def test_grouped_shap_is_the_sum_of_members() -> None:
    s = pd.DataFrame({"a": [0.1, -0.2], "b": [0.3, 0.1], "c": [0.5, 0.5]})
    g = groups.grouped(s, {"a": "g1", "b": "g1", "c": "g2"})
    assert np.allclose(g["g1"], [0.4, -0.1]) and np.allclose(g["g2"], [0.5, 0.5])


@pytest.mark.parametrize(("feature", "value", "expected"), [
    ("minutes_ratio_30_60", 0.38, "Listening time down 62% in the last 30 days"),
    ("minutes_ratio_30_60", 1.4, "Listening time up 40%"),
    ("minutes_ratio_30_60", float("nan"), "not comparable"),
    ("last_is_auto_renew", 0, "Plan is not on auto-renew"),
    ("last_payment_method", "38", "Pays with payment method 38"),
    ("days_since_last_listen", 12.0, "Last listened 12 days before the cutoff"),
    ("skip_rate_30d", 0.25, "Skips 25% of songs"),
])
def test_reason_statements(feature: str, value: object, expected: str) -> None:
    assert expected in reasons.statement(feature, value, TEMPLATES)


def test_reason_wording_is_associational_not_causal() -> None:
    up = reasons.reason("last_is_auto_renew", 0, 0.8, TEMPLATES)
    down = reasons.reason("last_is_auto_renew", 1, -0.8, TEMPLATES)
    assert up.endswith(reasons.RAISES) and down.endswith(reasons.LOWERS)
    for text in (up, down):
        assert "cause" not in text.lower() and "because" not in text.lower()


def test_top_drivers_rank_clusters_and_use_strongest_member() -> None:
    shap_df = pd.DataFrame({"m7": [0.30, -0.1], "m30": [0.20, -0.1], "pay": [0.40, 0.005],
                            "renew": [-0.5, 0.02]})
    values = pd.DataFrame({"m7": [10.0, 50.0], "m30": [40.0, 200.0], "pay": ["38", "41"],
                           "renew": [1.0, 0.0]})
    cluster_of = {"m7": "c1", "m30": "c1", "pay": "cat_pay", "renew": "c2"}
    names = {"c1": "m7 (+1)", "cat_pay": "pay", "c2": "renew"}
    group_of = dict.fromkeys(cluster_of, "g")
    d = reasons.top_drivers(shap_df, values, pd.Series(["u1", "u2"]), cluster_of, names,
                            group_of, TEMPLATES, top_n=5, min_contribution=0.01)
    u1 = d[d["msno"] == "u1"]
    assert list(u1["cluster"]) == ["m7 (+1)", "pay"]          # 0.50 then 0.40; renew lowers
    assert u1.iloc[0]["feature"] == "m7" and u1.iloc[0]["contribution"] == pytest.approx(0.5)
    u2 = d[d["msno"] == "u2"]
    assert list(u2["cluster"]) == ["renew"]                    # 0.005 is below the floor
