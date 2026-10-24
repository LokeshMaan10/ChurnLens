"""K-Means on scaled RFM values for a configured population, compared with RFM segments.

Only users matching `kmeans.population` (default: paid > 0 and listened on >= 1 day) are
clustered; everyone else gets an explicit rule-based label from `kmeans.excluded_groups`.
K is chosen by silhouette score (elbow reported alongside). Cluster numbers are assigned
by size on the reference snapshot (cluster 1 = largest), so they mean the same thing on
every run. Clusters are kept only if they add churn information beyond the RFM segments
within the clustered population (config: min_cramers_v_gain).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from churnlens.analysis.stats import chi2_independence
from churnlens.segmentation.rfm import segment_case_sql


def feature_matrix(raw: pd.DataFrame) -> np.ndarray:
    """log1p(recency), frequency, log1p(monetary): skewed values compressed before scaling."""
    return np.column_stack([np.log1p(raw["recency_days"]), raw["frequency_days"],
                            np.log1p(raw["monetary"])]).astype(float)


def _eval(raw: pd.DataFrame, sql: str) -> pd.Series:
    con = duckdb.connect()
    con.register("raw", raw)
    out = con.execute(f"SELECT {sql} AS v FROM raw").df()["v"]
    con.close()
    return out.set_axis(raw.index)


def in_population(raw: pd.DataFrame, predicate: str) -> pd.Series:
    """Boolean mask of users to cluster (DuckDB SQL predicate over raw RFM columns)."""
    return _eval(raw, f"COALESCE({predicate}, false)").astype(bool)


def excluded_labels(raw: pd.DataFrame, groups: list[dict[str, str]]) -> pd.Series:
    """Rule-based labels for users outside the population (first match wins)."""
    return _eval(raw, segment_case_sql(groups))


@dataclass
class KMeansResult:
    """Fitted scaler/model, size-ordered label map, and selection diagnostics."""

    scaler: StandardScaler
    model: KMeans
    k: int
    label_map: dict[int, int]      # raw KMeans label -> cluster number (1 = largest)
    diagnostics: pd.DataFrame


def select_and_fit(pop: pd.DataFrame, cfg: dict[str, Any]) -> KMeansResult:
    """Fit K-Means for each K on the population; choose K by highest silhouette (sampled)."""
    x = feature_matrix(pop)
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    rng = np.random.default_rng(cfg["seed"])
    sample = rng.choice(len(xs), size=min(cfg["silhouette_sample"], len(xs)), replace=False)
    rows, models = [], {}
    lo, hi = cfg["k_range"]
    for k in range(lo, hi + 1):
        m = KMeans(n_clusters=k, n_init=cfg["n_init"], random_state=cfg["seed"]).fit(xs)
        models[k] = m
        sil = silhouette_score(xs[sample], m.labels_[sample], random_state=cfg["seed"])
        rows.append({"k": k, "inertia": float(m.inertia_), "silhouette": float(sil)})
    diag = pd.DataFrame(rows)
    best = int(diag.loc[diag["silhouette"].idxmax(), "k"])
    model = models[best]
    # Size order (ties broken by raw label) gives run-to-run stable cluster numbers.
    sizes = np.bincount(model.labels_, minlength=best)
    order = sorted(range(best), key=lambda lab: (-sizes[lab], lab))
    label_map = {raw_label: rank + 1 for rank, raw_label in enumerate(order)}
    return KMeansResult(scaler, model, best, label_map, diag)


def assign(result: KMeansResult, raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """'cluster N' for population users (reference-fitted model); exclusion label otherwise."""
    mask = in_population(raw, cfg["population"])
    labels = excluded_labels(raw, cfg["excluded_groups"]).astype(object)
    if mask.any():
        raw_labels = result.model.predict(result.scaler.transform(feature_matrix(raw[mask])))
        labels[mask] = [f"cluster {result.label_map[int(lab)]}" for lab in raw_labels]
    return labels


def _v(labels: pd.Series, churn: pd.Series) -> float:
    counts = [(int(churn[labels == g].sum()), int((labels == g).sum())) for g in labels.unique()]
    return chi2_independence(counts)[2]


def compare(segments: pd.Series, clusters: pd.Series, churn: pd.Series,
            min_gain: float) -> dict[str, Any]:
    """Does clustering add churn information beyond the RFM segments (same users)?"""
    v_seg = _v(segments, churn)
    v_clu = _v(clusters.astype(str), churn)
    v_both = _v(segments + " | " + clusters.astype(str), churn)
    gain = v_both - v_seg
    return {"cramers_v_segments": v_seg, "cramers_v_clusters": v_clu,
            "cramers_v_segment_x_cluster": v_both, "gain": gain,
            "adjusted_rand_index": float(adjusted_rand_score(segments, clusters)),
            "keep": bool(gain >= min_gain)}
