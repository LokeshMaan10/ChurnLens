"""SIMULATED experiments: randomized assignment, fake treatment effects, uplift demo.

Everything here is a simulation. Outcomes are drawn from the model's calibrated churn
probabilities and FAKE treatment effects from configs/experiments.yaml; the results
demonstrate the analysis tooling and must never be read as evidence about real effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

SIMULATED = "SIMULATED"


def assign(n: int, treatment_share: float, rng: np.random.Generator) -> np.ndarray:
    """Randomized assignment: exactly round(n * share) treated, positions shuffled."""
    n_t = int(round(n * treatment_share))
    arm = np.zeros(n, dtype=bool)
    arm[:n_t] = True
    rng.shuffle(arm)
    return arm


def fake_effect(manual_renew: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """FAKE relative churn reduction per user (heterogeneous by auto-renew status)."""
    e = cfg["fake_effect"]
    return np.where(manual_renew, e["manual_renew_relative_reduction"],
                    e["base_relative_reduction"])


def simulate_outcomes(p: np.ndarray, treated: np.ndarray, effect: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """SIMULATED churn: Bernoulli(p) for control, Bernoulli(p * (1 - effect)) if treated."""
    prob = np.where(treated, p * (1 - effect), p)
    return (rng.uniform(size=len(p)) < prob).astype(int)


def t_learner(X: pd.DataFrame, treated: np.ndarray, y: np.ndarray, test: np.ndarray,
              cfg: dict[str, Any], seed: int) -> np.ndarray:
    """T-learner: separate churn models for treated and control (train rows only);
    predicted uplift = P(churn | control) - P(churn | treated) on the test rows."""
    params = {"n_estimators": cfg["n_estimators"], "learning_rate": cfg["learning_rate"],
              "num_leaves": cfg["num_leaves"], "min_child_samples": cfg["min_child_samples"],
              "random_state": seed, "verbose": -1,
              "deterministic": True, "force_row_wise": True, "n_jobs": 4}
    train = ~test
    m_t = LGBMClassifier(**params).fit(X[train & treated], y[train & treated])
    m_c = LGBMClassifier(**params).fit(X[train & ~treated], y[train & ~treated])
    Xt = X[test]
    return m_c.predict_proba(Xt)[:, 1] - m_t.predict_proba(Xt)[:, 1]


def uplift_deciles(pred_uplift: np.ndarray, true_uplift: np.ndarray, treated: np.ndarray,
                   y: np.ndarray, bins: int) -> pd.DataFrame:
    """By predicted-uplift decile (1 = highest): mean predicted vs TRUE simulated uplift,
    and the uplift observed in the simulated outcomes (control rate - treated rate)."""
    order = np.argsort(-pred_uplift, kind="stable")
    rows = []
    for i, idx in enumerate(np.array_split(order, bins), 1):
        t, c = treated[idx], ~treated[idx]
        obs = (y[idx][c].mean() - y[idx][t].mean()) if t.any() and c.any() else np.nan
        rows.append({"decile": i, "users": len(idx),
                     "predicted_uplift": float(pred_uplift[idx].mean()),
                     "true_uplift": float(true_uplift[idx].mean()),
                     "observed_uplift": float(obs)})
    return pd.DataFrame(rows)
