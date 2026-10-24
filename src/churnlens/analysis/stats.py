"""Small statistical helpers for descriptive analysis (rates, intervals, tests)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Rate:
    """A proportion with its Wilson score interval."""

    k: int
    n: int
    rate: float
    lo: float
    hi: float


def wilson(k: int, n: int, alpha: float) -> Rate:
    """Wilson score interval for k successes in n trials (well-behaved for small n and rates)."""
    if n == 0:
        return Rate(k, n, float("nan"), float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # The interval is exactly [0, ...] at k = 0 and [..., 1] at k = n; pin them so float
    # rounding (e.g. 7e-18) does not leak into reports.
    lo = 0.0 if k == 0 else max(0.0, float(centre - half))
    hi = 1.0 if k == n else min(1.0, float(centre + half))
    return Rate(k, n, p, lo, hi)


def chi2_independence(counts: list[tuple[int, int]]) -> tuple[float, float, float]:
    """Chi-square test of churn vs group. counts = [(churners, n), ...].

    Returns (chi2, p_value, Cramér's V). V measures association strength (0..1).
    """
    table = np.array([[k, n - k] for k, n in counts], dtype=float)
    table = table[table.sum(axis=1) > 0]
    if table.shape[0] < 2 or (table.sum(axis=0) == 0).any():
        return float("nan"), float("nan"), float("nan")
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    v = math.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))
    return float(chi2), float(p), v


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-sided two-proportion z-test with pooled variance. Returns (z, p_value)."""
    if min(n1, n2) == 0:
        return float("nan"), float("nan")
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (k1 / n1 - k2 / n2) / se
    return z, float(2 * stats.norm.sf(abs(z)))


def fmt_p(p: float) -> str:
    """Compact p-value string."""
    if math.isnan(p):
        return "—"
    return "<1e-300" if p == 0 else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")
