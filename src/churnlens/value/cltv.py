"""Customer lifetime value, two ways, plus a revenue backtest.

CLTV = expected discounted margin over the horizon IF the customer renews at the
current expiry. With s_j = P(still subscribed j months after renewing), monthly
margin m and monthly discount rate d:
    CLTV = m * sum_{k=1..H} s_{k-1} / (1 + d)^k
  formula:  s_j = r^j, r = renewal rate of the user's segment in the fit snapshot
  survival: s_j from the Cox model, conditional on the current spell's age
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def monthly_discount(annual: float) -> float:
    """Annual discount rate -> equivalent monthly rate."""
    return (1 + annual) ** (1 / 12) - 1


def monthly_price(features: pd.DataFrame, median_price: float, days_per_month: int) -> pd.Series:
    """Monthly price from the latest plan: paid per day x 30; free or list-price-0 plans
    fall back to list price per day, then to the median paid monthly price."""
    days = features["last_plan_days"].where(features["last_plan_days"] > 0)
    paid = features["last_amount_paid"] / days * days_per_month
    listed = features["last_list_price"] / days * days_per_month
    price = paid.where(paid > 0, listed.where(listed > 0))
    return price.fillna(median_price)


def plan_bucket(plan_days: pd.Series, buckets: list[dict[str, Any]]) -> pd.Series:
    """Plan-length bucket name (first bucket whose max_days covers the value)."""
    out = pd.Series("no plan", index=plan_days.index, dtype=object)
    remaining = plan_days.notna()
    for b in buckets:
        hit = remaining & (plan_days <= b["max_days"])
        out[hit] = b["name"]
        remaining &= ~hit
    return out


def segment_key(features: pd.DataFrame, buckets: list[dict[str, Any]]) -> pd.Series:
    """Formula segment: auto-renew status x plan-length bucket."""
    ar = features["last_is_auto_renew"].map({1: "auto-renew", 0: "manual"}).fillna("none")
    return ar + " / " + plan_bucket(features["last_plan_days"], buckets)


def segment_renewal_rates(segments: pd.Series, churn: np.ndarray, min_users: int
                          ) -> pd.DataFrame:
    """Renewal rate (1 - churn rate) per segment from the fit snapshot's labels; segments
    under `min_users` use the overall rate instead."""
    df = pd.DataFrame({"segment": segments.to_numpy(), "churn": churn})
    out = df.groupby("segment")["churn"].agg(["size", "mean"]).rename(
        columns={"size": "users", "mean": "churn_rate"})
    small = out["users"] < min_users
    out["renewal_rate"] = np.where(small, 1 - float(np.mean(churn)), 1 - out["churn_rate"])
    out["uses_overall_rate"] = small
    return out.reset_index()


def cltv_from_survival(margin_month: np.ndarray, s: np.ndarray, d: float) -> np.ndarray:
    """m * sum_k s_{k-1} / (1+d)^k for each row of the survival matrix s (n x H)."""
    horizon = s.shape[1]
    disc = 1 / (1 + d) ** np.arange(1, horizon + 1)
    return margin_month * (s * disc).sum(axis=1)


def formula_survival(renewal: np.ndarray, horizon: int) -> np.ndarray:
    """s_j = r^j for j = 0..H-1."""
    return renewal[:, None] ** np.arange(horizon)[None, :]


def backtest(realized: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Error and ranking agreement of predicted vs realized revenue."""
    from scipy.stats import spearmanr
    err = predicted - realized
    return {"users": int(len(realized)), "realized_total": float(realized.sum()),
            "predicted_total": float(predicted.sum()),
            "total_ratio": float(predicted.sum() / realized.sum()) if realized.sum() else np.nan,
            "mae": float(np.abs(err).mean()), "bias": float(err.mean()),
            "spearman": float(spearmanr(predicted, realized).statistic)}
