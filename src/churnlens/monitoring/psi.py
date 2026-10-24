"""Population Stability Index, hand-written.

PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i) over bins, where ref_i / cur_i are the
shares of the reference and current samples in bin i. Numeric features use quantile bins
fitted on the reference (duplicate edges merged, so piled-up values form one bin); missing
values always get their own bin. Categorical features use the reference's categories plus
an "other" bin for unseen ones. Empty bins are floored at `eps` so the log is finite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_BIN = "__missing__"
OTHER_BIN = "__other__"


def _psi(ref_share: np.ndarray, cur_share: np.ndarray, eps: float) -> float:
    r = np.maximum(ref_share, eps)
    c = np.maximum(cur_share, eps)
    return float(np.sum((c - r) * np.log(c / r)))


def numeric_bins(ref: pd.Series, bins: int) -> np.ndarray:
    """Interior quantile edges from the reference (unique, so ties don't create empty bins)."""
    vals = ref.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.array([])
    qs = np.quantile(vals, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(qs)


def bin_shares(s: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Shares per bin: [missing, (-inf, e0], (e0, e1], ..., (e_k, inf)]."""
    vals = s.to_numpy(dtype=float)
    miss = np.isnan(vals)
    idx = np.searchsorted(edges, vals[~miss], side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    total = len(vals)
    return np.concatenate([[miss.sum() / total], counts / total]) if total else counts


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int, eps: float) -> float:
    edges = numeric_bins(ref, bins)
    return _psi(bin_shares(ref, edges), bin_shares(cur, edges), eps)


def psi_categorical(ref: pd.Series, cur: pd.Series, eps: float) -> float:
    r = ref.astype(object).where(ref.notna(), MISSING_BIN).astype(str)
    c = cur.astype(object).where(cur.notna(), MISSING_BIN).astype(str)
    cats = sorted(set(r))
    c = c.where(c.isin(cats), OTHER_BIN)
    levels = [*cats, OTHER_BIN]
    rs = r.value_counts(normalize=True).reindex(levels, fill_value=0.0).to_numpy()
    cs = c.value_counts(normalize=True).reindex(levels, fill_value=0.0).to_numpy()
    return _psi(rs, cs, eps)


def band(value: float, moderate: float, major: float) -> str:
    """'stable' / 'moderate' / 'major' using the configured PSI thresholds."""
    return "major" if value > major else "moderate" if value > moderate else "stable"
