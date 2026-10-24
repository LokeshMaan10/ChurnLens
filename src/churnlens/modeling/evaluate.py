"""Evaluation metrics: discrimination, calibration, ranking (top-K, lift, gains), fairness.

Two inputs are kept apart on purpose:
  score  the raw model output, used for anything that only needs an ORDER (PR-AUC,
         ROC-AUC, top-K, lift, gains). Calibration can create ties (isotonic is a step
         function) that blur the ranking, so ranking never uses calibrated values.
  prob   the calibrated probability, used for anything that needs the VALUE (log loss,
         Brier, calibration tables, mean prediction, threshold decisions).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from churnlens.modeling.train import PROB_CLIP


def metrics(y: np.ndarray, score: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Ranking metrics from the raw score; probability metrics from the calibrated prob."""
    return {"pr_auc": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)),
            "log_loss": float(log_loss(y, np.clip(prob, PROB_CLIP, 1 - PROB_CLIP))),
            "brier": float(brier_score_loss(y, prob)),
            "base_rate": float(y.mean()), "mean_predicted": float(prob.mean())}


def _rank(p: np.ndarray) -> np.ndarray:
    """Indices from highest to lowest probability; ties broken by position (stable)."""
    return np.argsort(-p, kind="stable")


def top_k(y: np.ndarray, score: np.ndarray, pcts: list[float]) -> pd.DataFrame:
    """Precision, recall and lift when contacting the top K% by raw risk score."""
    order, n, pos = _rank(score), len(y), y.sum()
    rows = []
    for k in pcts:
        m = max(1, int(round(n * k / 100)))
        hits = int(y[order[:m]].sum())
        rows.append({"top_pct": k, "contacted": m, "churners_caught": hits,
                     "precision": hits / m, "recall": hits / pos if pos else np.nan,
                     "lift": (hits / m) / y.mean() if y.mean() else np.nan})
    return pd.DataFrame(rows)


def lift_table(y: np.ndarray, score: np.ndarray, prob: np.ndarray, bins: int) -> pd.DataFrame:
    """Equal-size bins ranked by raw score (1 = highest risk): churn rate, lift, cumulative
    gains, and the mean calibrated probability in each bin."""
    order = _rank(score)
    p = prob
    groups = np.array_split(order, bins)
    total = y.sum()
    rows, cum = [], 0
    for i, idx in enumerate(groups, 1):
        hits = int(y[idx].sum())
        cum += hits
        rows.append({"bin": i, "users": len(idx), "churners": hits,
                     "churn_rate": hits / len(idx), "mean_predicted": float(p[idx].mean()),
                     "lift": (hits / len(idx)) / y.mean(),
                     "cumulative_population": sum(len(g) for g in groups[:i]) / len(y),
                     "cumulative_gain": cum / total if total else np.nan})
    return pd.DataFrame(rows)


def calibration_table(y: np.ndarray, score: np.ndarray, p: np.ndarray,
                      bins: int) -> pd.DataFrame:
    """Equal-size bins ordered by raw score (so tied calibrated values never decide bin
    membership): mean calibrated probability vs observed churn."""
    order = np.argsort(score, kind="stable")
    rows = []
    for i, idx in enumerate(np.array_split(order, bins), 1):
        rows.append({"bin": i, "users": len(idx), "mean_predicted": float(p[idx].mean()),
                     "observed_rate": float(y[idx].mean())})
    return pd.DataFrame(rows)


def confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """Confusion matrix and rates at a threshold."""
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": threshold, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "fpr": fp / (fp + tn) if fp + tn else np.nan,
            "contacted_share": float(pred.mean())}


def fairness_table(groups: pd.Series, y: np.ndarray, p: np.ndarray, threshold: float,
                   min_n: int) -> pd.DataFrame:
    """Error rates by group at the chosen threshold (groups below min_n are flagged)."""
    rows = []
    for g in sorted(groups.unique(), key=str):
        m = (groups == g).to_numpy()
        yt, pt = y[m], p[m]
        c = confusion(yt, pt, threshold)
        rows.append({"group": g, "users": int(m.sum()), "churn_rate": float(yt.mean()),
                     "mean_predicted": float(pt.mean()), "flagged_share": c["contacted_share"],
                     "recall": c["recall"], "fnr": 1 - c["recall"] if c["recall"] == c["recall"]
                     else np.nan, "fpr": c["fpr"], "precision": c["precision"],
                     "small_group": bool(m.sum() < min_n)})
    return pd.DataFrame(rows)
