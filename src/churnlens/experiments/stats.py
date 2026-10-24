"""Experiment statistics: sample size / power, two-proportion tests, sample-ratio checks.

Normal-approximation formulas (two-sided). Treatment and control sizes may differ:
`ratio` = n_treatment / n_control.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy import stats


def sample_size(p_control: float, p_treatment: float, alpha: float, power: float,
                ratio: float = 1.0) -> tuple[int, int]:
    """(n_control, n_treatment) to detect p_control vs p_treatment with a two-sided test.

    n_c = [z_a * sqrt(pbar*qbar*(1 + 1/k)) + z_b * sqrt(p_c*q_c + p_t*q_t/k)]^2 / delta^2,
    n_t = k * n_c, with pbar the size-weighted pooled rate (Fleiss, without correction).
    """
    if p_control == p_treatment:
        raise ValueError("p_control and p_treatment must differ")
    k = ratio
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p_control + k * p_treatment) / (1 + k)
    num = (za * math.sqrt(pbar * (1 - pbar) * (1 + 1 / k))
           + zb * math.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment) / k))
    n_c = math.ceil(num**2 / (p_control - p_treatment) ** 2)
    return n_c, math.ceil(k * n_c)


def achieved_power(p_control: float, p_treatment: float, n_control: int, n_treatment: int,
                   alpha: float) -> float:
    """Power of a two-sided two-proportion z-test for the given sizes."""
    za = stats.norm.ppf(1 - alpha / 2)
    k = n_treatment / n_control
    pbar = (p_control + k * p_treatment) / (1 + k)
    se0 = math.sqrt(pbar * (1 - pbar) * (1 / n_control + 1 / n_treatment))
    se1 = math.sqrt(p_control * (1 - p_control) / n_control
                    + p_treatment * (1 - p_treatment) / n_treatment)
    delta = abs(p_control - p_treatment)
    return float(stats.norm.cdf((delta - za * se0) / se1)
                 + stats.norm.cdf((-delta - za * se0) / se1))


@dataclass
class TwoProportionResult:
    """Outcome of a two-proportion comparison (treatment minus control)."""

    n_control: int
    n_treatment: int
    rate_control: float
    rate_treatment: float
    difference: float          # treatment - control (negative = fewer churners)
    ci_low: float
    ci_high: float
    relative_change: float
    z: float
    p_value: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def two_proportion_test(x_control: int, n_control: int, x_treatment: int, n_treatment: int,
                        alpha: float) -> TwoProportionResult:
    """Pooled z-test for H0: equal rates; unpooled Wald CI for the difference."""
    pc, pt = x_control / n_control, x_treatment / n_treatment
    pooled = (x_control + x_treatment) / (n_control + n_treatment)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    z = (pt - pc) / se0 if se0 else float("nan")
    p = float(2 * stats.norm.sf(abs(z))) if se0 else float("nan")
    se = math.sqrt(pc * (1 - pc) / n_control + pt * (1 - pt) / n_treatment)
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = pt - pc
    return TwoProportionResult(n_control, n_treatment, pc, pt, diff, diff - zc * se,
                               diff + zc * se, diff / pc if pc else float("nan"), z, p)


def srm_check(n_control: int, n_treatment: int, expected_treatment_share: float,
              alpha: float) -> dict[str, float | bool]:
    """Chi-square goodness-of-fit test of the observed split vs the planned split."""
    n = n_control + n_treatment
    expected = [n * (1 - expected_treatment_share), n * expected_treatment_share]
    chi2, p = stats.chisquare([n_control, n_treatment], f_exp=expected)
    return {"observed_treatment_share": n_treatment / n, "expected_treatment_share":
            expected_treatment_share, "chi2": float(chi2), "p_value": float(p),
            "mismatch": bool(p < alpha)}
