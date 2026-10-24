"""Exact TreeSHAP for the trained LightGBM pipeline, mapped back to model features.

The model sees one-hot columns for categorical features. SHAP values are additive, so
a categorical feature's contribution is the sum over its one-hot columns; numeric
features map one-to-one. Contributions are in raw-score units (log-odds of the
class-weighted model). The additivity check (base + sum = raw score) is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

ADDITIVITY_TOLERANCE = 1e-6
CHUNK_ROWS = 10_000


@dataclass
class Explanation:
    """Feature-level SHAP values for a set of users."""

    shap: pd.DataFrame          # rows = users, columns = model features (log-odds)
    base_value: float           # expected raw score (log-odds)
    raw_score: np.ndarray       # model raw score per user (log-odds)


def column_to_feature(transformed: list[str], numeric: list[str],
                      categorical: list[str]) -> list[str]:
    """Map each transformed column to the model feature it came from.

    Numeric columns keep their names; one-hot columns are named `<feature>_<category>`
    and are matched to the longest categorical feature name that prefixes them.
    """
    num = set(numeric)
    by_length = sorted(categorical, key=len, reverse=True)
    out = []
    for col in transformed:
        if col in num:
            out.append(col)
            continue
        match = next((f for f in by_length if col.startswith(f + "_")), None)
        if match is None:
            raise ValueError(f"cannot map transformed column {col!r} to a feature")
        out.append(match)
    return out


def explain(pipeline: Pipeline, X: pd.DataFrame, numeric: list[str],
            categorical: list[str], log=print) -> Explanation:
    """TreeSHAP in chunks; aggregate one-hot columns per feature; check additivity."""
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    names = list(prep.get_feature_names_out())
    owner = np.array(column_to_feature(names, numeric, categorical))
    features = numeric + categorical
    parts, raws = [], []
    for start in range(0, len(X), CHUNK_ROWS):
        xt = prep.transform(X.iloc[start:start + CHUNK_ROWS])
        sv = np.asarray(explainer.shap_values(xt))
        agg = np.column_stack([sv[:, owner == f].sum(axis=1) for f in features])
        parts.append(agg)
        raws.append(clf.predict(xt, raw_score=True))
        log(f"  shap rows {min(start + CHUNK_ROWS, len(X)):,}/{len(X):,}")
    values = np.vstack(parts)
    raw = np.concatenate(raws)
    base = float(np.ravel(explainer.expected_value)[0])
    err = np.abs(values.sum(axis=1) + base - raw).max()
    if err > ADDITIVITY_TOLERANCE:
        raise AssertionError(f"SHAP additivity violated: max error {err:.2e}")
    return Explanation(pd.DataFrame(values, columns=features, index=X.index), base, raw)
