"""Tuning, out-of-fold predictions and probability calibration (training snapshot only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from churnlens.modeling.data import param_distributions

PROB_CLIP = 1e-6   # keeps logit / log loss finite at 0 and 1


def folds(n: int, seed: int) -> StratifiedKFold:
    """Stratified, shuffled folds with a fixed seed."""
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)


@dataclass
class TuneResult:
    """Best parameters and the CV score summary for one model family."""

    best_params: dict[str, Any]
    best_score: float
    cv_table: pd.DataFrame


def tune(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray, spec: dict[str, Any],
         cfg: dict[str, Any]) -> TuneResult:
    """Randomized search over the configured space, scored by CV PR-AUC."""
    search = RandomizedSearchCV(
        estimator, param_distributions(spec["params"]), n_iter=spec["n_iter"],
        scoring=cfg["cv"]["scoring"], cv=folds(cfg["cv"]["folds"], cfg["seed"]),
        random_state=cfg["seed"], refit=False, n_jobs=1)
    search.fit(X, y)
    res = pd.DataFrame(search.cv_results_)
    table = res[["params", "mean_test_score", "std_test_score", "mean_fit_time"]] \
        .sort_values("mean_test_score", ascending=False)
    return TuneResult(dict(search.best_params_), float(search.best_score_), table)


def oof_predictions(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray,
                    cfg: dict[str, Any]) -> np.ndarray:
    """Out-of-fold churn probabilities (each row scored by a model that never saw it)."""
    return cross_val_predict(estimator, X, y, cv=folds(cfg["cv"]["folds"], cfg["seed"]),
                             method="predict_proba")[:, 1]


class Calibrator:
    """Maps raw model probabilities to calibrated ones (isotonic or Platt/sigmoid)."""

    def __init__(self, method: str) -> None:
        if method not in {"isotonic", "sigmoid"}:
            raise ValueError(f"unknown calibration method {method}")
        self.method = method
        self.model: Any = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        else:
            self.model = LogisticRegression().fit(self._logit(p), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return self.model.predict(p)
        return self.model.predict_proba(self._logit(p))[:, 1]


def cross_calibrate(p: np.ndarray, y: np.ndarray, method: str, n_folds: int,
                    seed: int) -> np.ndarray:
    """Calibrated OOF probabilities: each fold calibrated by a map fit on the other folds."""
    out = np.empty_like(p, dtype=float)
    for tr, te in folds(n_folds, seed).split(p.reshape(-1, 1), y):
        out[te] = Calibrator(method).fit(p[tr], y[tr]).predict(p[te])
    return out


def pick_method(briers: dict[str, float], preference: list[str], tolerance: float) -> str:
    """Lowest Brier wins; methods within `tolerance` of the best are tied, and ties go to
    the earliest method in `preference`."""
    best = min(briers.values())
    tied = [m for m in briers if briers[m] - best <= tolerance]
    ranked = sorted(tied, key=lambda m: preference.index(m) if m in preference
                    else len(preference))
    return ranked[0]


def choose_calibration(p: np.ndarray, y: np.ndarray, cfg: dict[str, Any]
                       ) -> tuple[str, pd.DataFrame, np.ndarray]:
    """Pick the calibration method by cross-validated Brier score (ties -> preference).

    Returns (method, comparison table, cross-calibrated OOF probabilities of the winner).
    """
    c = cfg["calibration"]
    rows, cal = [], {}
    rows.append({"method": "none (raw, class-weighted)", "brier": brier_score_loss(y, p),
                 "log_loss": log_loss(y, np.clip(p, PROB_CLIP, 1 - PROB_CLIP)),
                 "mean_predicted": float(p.mean())})
    for m in c["methods"]:
        cal[m] = cross_calibrate(p, y, m, c["folds"], cfg["seed"])
        rows.append({"method": m, "brier": brier_score_loss(y, cal[m]),
                     "log_loss": log_loss(y, np.clip(cal[m], PROB_CLIP, 1 - PROB_CLIP)),
                     "mean_predicted": float(cal[m].mean())})
    table = pd.DataFrame(rows)
    best = pick_method({m: brier_score_loss(y, cal[m]) for m in c["methods"]},
                       c["preference"], float(c["brier_tie_tolerance"]))
    return best, table, cal[best]


def fit_final(estimator: Pipeline, params: dict[str, Any], X: pd.DataFrame,
              y: np.ndarray) -> Pipeline:
    """Refit the tuned pipeline on the whole training snapshot."""
    return clone(estimator).set_params(**params).fit(X, y)
