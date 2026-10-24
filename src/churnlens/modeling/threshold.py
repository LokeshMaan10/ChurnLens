"""Contact threshold chosen by expected business cost (costs from configs/costs.yaml)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def threshold_grid(step: float) -> np.ndarray:
    """Candidate thresholds 0..1 in `step` increments (rounded to avoid float drift)."""
    return np.round(np.arange(0.0, 1.0 + step / 2, step), 6)


def breakeven(costs: dict[str, Any]) -> float:
    """Probability above which contacting is worth it: offer / (save_rate x churn_loss)."""
    return costs["offer_cost"] / (costs["save_rate"] * costs["churn_loss"])


def realized_cost(y: np.ndarray, contact: np.ndarray, costs: dict[str, Any]) -> float:
    """Total cost of a contact policy given actual outcomes.

    Contacted: offer cost, plus the loss for churners the action fails to save.
    Not contacted: the full loss for every churner.
    """
    offer, loss, save = costs["offer_cost"], costs["churn_loss"], costs["save_rate"]
    contacted = contact.astype(bool)
    return float(contacted.sum() * offer
                 + (y[contacted] == 1).sum() * (1 - save) * loss
                 + (y[~contacted] == 1).sum() * loss)


def cost_curve(y: np.ndarray, p: np.ndarray, costs: dict[str, Any],
               step: float) -> pd.DataFrame:
    """Realized cost per user at each threshold of the grid."""
    rows = [{"threshold": t, "contacted_share": float((p >= t).mean()),
             "cost_per_user": realized_cost(y, p >= t, costs) / len(y)}
            for t in threshold_grid(step)]
    return pd.DataFrame(rows)


def choose_threshold(y: np.ndarray, p: np.ndarray, costs: dict[str, Any],
                     step: float) -> float:
    """Threshold minimising realized cost on the training snapshot's calibrated OOF
    probabilities (ties: the higher threshold, i.e. fewer contacts)."""
    curve = cost_curve(y, p, costs, step)
    best = curve["cost_per_user"].min()
    return float(curve.loc[np.isclose(curve["cost_per_user"], best), "threshold"].max())


def policy_summary(y: np.ndarray, p: np.ndarray, threshold: float,
                   costs: dict[str, Any]) -> pd.DataFrame:
    """Cost of the model policy vs contacting nobody / everybody."""
    n = len(y)
    rows = []
    for name, contact in (("model (contact if p >= threshold)", p >= threshold),
                          ("contact nobody", np.zeros(n, bool)),
                          ("contact everybody", np.ones(n, bool))):
        rows.append({"policy": name, "contacted": int(contact.sum()),
                     "contacted_share": float(contact.mean()),
                     "churners_contacted": int((y[contact] == 1).sum()),
                     "total_cost": realized_cost(y, contact, costs),
                     "cost_per_user": realized_cost(y, contact, costs) / n})
    return pd.DataFrame(rows)
