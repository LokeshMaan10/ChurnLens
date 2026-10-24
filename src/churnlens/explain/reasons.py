"""Plain-language reason codes and per-customer top drivers.

A driver is a correlation cluster whose summed SHAP raises this customer's risk. It is
described by the cluster member with the largest contribution for that customer, using
a neutral template from configs/reason_codes.yaml plus association wording. Reason
codes say what is associated with the model's prediction; they are not causes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

RAISES = "associated with higher predicted churn risk"
LOWERS = "associated with lower predicted churn risk"
UNCHANGED_BAND = 0.005   # |ratio - 1| below this reads as "unchanged"


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or v == "missing"


def format_value(value: Any, fmt: str) -> str:
    """Render a feature value for a reason template."""
    if _missing(value):
        return "no data"
    if fmt == "pct_change":
        change = float(value) - 1.0
        if abs(change) < UNCHANGED_BAND:
            return "unchanged"
        return f"{'down' if change < 0 else 'up'} {abs(change) * 100:.0f}%"
    if fmt in ("days", "count"):
        return f"{float(value):,.0f}"
    if fmt == "minutes":
        return f"{float(value):,.0f}"
    if fmt == "ntd":
        return f"{float(value):,.0f} NTD"
    if fmt == "share":
        return f"{float(value) * 100:.0f}%"
    if fmt == "number":
        return f"{float(value):,.2f}"
    return str(value)


def humanize(feature: str) -> str:
    """Fallback label: snake_case -> words."""
    return feature.replace("_", " ").capitalize()


def statement(feature: str, value: Any, templates: dict[str, dict[str, str]]) -> str:
    """Neutral sentence stating the customer's value for one feature."""
    t = templates.get(feature)
    if t is None:
        fmt = "number" if isinstance(value, (int, float)) else "category"
        return f"{humanize(feature)}: {format_value(value, fmt)}"
    if t["format"] == "flag":
        if _missing(value):
            return f"{humanize(feature)}: no data"
        return t["when_1"] if float(value) >= 1 else t["when_0"]
    v = format_value(value, t["format"])
    if v == "no data" and t["format"] == "pct_change":
        return t["text"].replace("{value}", "not comparable (no earlier listening)")
    text = t["text"].replace("{value}", v)
    # Singular units read naturally ("1 day", not "1 days").
    return text.replace(" 1 days", " 1 day").replace(" 1 times", " once") \
        .replace("Changed plan 1 times", "Changed plan once")


def reason(feature: str, value: Any, contribution: float,
           templates: dict[str, dict[str, str]]) -> str:
    """Full reason code: statement + association direction."""
    return f"{statement(feature, value, templates)} — {RAISES if contribution > 0 else LOWERS}"


def top_drivers(shap_df: pd.DataFrame, values: pd.DataFrame, msno: pd.Series,
                cluster_of: dict[str, str], cluster_names: dict[str, str],
                group_of: dict[str, str], templates: dict[str, dict[str, str]],
                top_n: int, min_contribution: float) -> pd.DataFrame:
    """Long table: up to `top_n` risk-raising clusters per user, ranked by summed SHAP."""
    features = list(shap_df.columns)
    clusters = sorted({cluster_of[f] for f in features})
    members = {c: [f for f in features if cluster_of[f] == c] for c in clusters}
    sv = shap_df.to_numpy()
    col = {f: i for i, f in enumerate(features)}
    csum = np.column_stack([sv[:, [col[f] for f in members[c]]].sum(axis=1) for c in clusters])
    # Strongest member of each cluster for each user (by that member's own SHAP).
    lead = np.column_stack([np.array([col[f] for f in members[c]])[
        sv[:, [col[f] for f in members[c]]].argmax(axis=1)] for c in clusters])
    order = np.argsort(-csum, axis=1, kind="stable")[:, :top_n]
    rows = []
    vals = values.reset_index(drop=True)
    ids = msno.reset_index(drop=True)
    for i in range(len(sv)):
        for rank, j in enumerate(order[i], 1):
            contrib = float(csum[i, j])
            if contrib < min_contribution:
                break
            f = features[lead[i, j]]
            v = vals.at[i, f]
            rows.append({"msno": ids.iat[i], "rank": rank, "cluster": cluster_names[clusters[j]],
                         "feature": f, "group": group_of[f],
                         "value": None if _missing(v) else str(v),
                         "contribution": round(contrib, 6),
                         "feature_contribution": round(float(sv[i, lead[i, j]]), 6),
                         "reason": reason(f, v, contrib, templates)})
    return pd.DataFrame(rows)
