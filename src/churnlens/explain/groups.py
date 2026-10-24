"""Feature groups for importance and drivers: registry groups and correlation clusters.

Correlated features split credit between them (minutes_7d, minutes_14d and minutes_30d
carry overlapping information), so per-feature importance understates each one. Summing
SHAP within a group gives the group's total contribution exactly (additivity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def correlation_clusters(X: pd.DataFrame, numeric: list[str], categorical: list[str],
                         abs_rho: float, sample: int, seed: int) -> dict[str, str]:
    """feature -> cluster id. Numeric features with |Spearman rho| > abs_rho (average
    linkage on 1 - |rho|) share a cluster; each categorical feature is its own cluster."""
    rows = X[numeric].sample(n=min(sample, len(X)), random_state=seed)
    rho = rows.rank().corr().abs().fillna(0.0).to_numpy()   # Spearman = Pearson on ranks
    np.fill_diagonal(rho, 1.0)
    dist = squareform(1.0 - rho, checks=False)
    labels = fcluster(linkage(dist, method="average"), t=1.0 - abs_rho, criterion="distance")
    out = {f: f"c{int(lab):02d}" for f, lab in zip(numeric, labels, strict=True)}
    out.update({f: f"cat_{f}" for f in categorical})
    return out


def name_clusters(cluster_of: dict[str, str], importance: pd.Series) -> dict[str, str]:
    """Readable cluster names: the most important member, plus '+N' for the others."""
    members: dict[str, list[str]] = {}
    for f, c in cluster_of.items():
        members.setdefault(c, []).append(f)
    names = {}
    for c, fs in members.items():
        lead = max(fs, key=lambda f: (importance.get(f, 0.0), f))
        names[c] = lead if len(fs) == 1 else f"{lead} (+{len(fs) - 1})"
    return names


def grouped(shap_df: pd.DataFrame, group_of: dict[str, str]) -> pd.DataFrame:
    """Per-user SHAP summed within each group (columns = groups)."""
    groups = pd.Series(group_of)
    return shap_df.T.groupby(groups.reindex(shap_df.columns).to_numpy()).sum().T


def importance(shap_df: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per column, descending."""
    return shap_df.abs().mean().sort_values(ascending=False)
