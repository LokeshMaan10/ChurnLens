"""Phase 10: sample size, power, two-proportion test, SRM, simulation helpers."""

import numpy as np
import pytest

from churnlens.experiments import simulate
from churnlens.experiments import stats as xs


def test_sample_size_matches_textbook_value() -> None:
    # Classic example: 0.50 vs 0.60, alpha 0.05 two-sided, power 0.80 -> ~388 per group.
    n_c, n_t = xs.sample_size(0.5, 0.6, 0.05, 0.8)
    assert n_c == n_t and 385 <= n_c <= 390


def test_power_is_consistent_with_sample_size() -> None:
    n_c, n_t = xs.sample_size(0.10, 0.08, 0.05, 0.8, ratio=4.0)
    assert n_t == pytest.approx(4 * n_c, abs=1)
    assert xs.achieved_power(0.10, 0.08, n_c, n_t, 0.05) == pytest.approx(0.80, abs=0.01)


def test_two_proportion_test() -> None:
    r = xs.two_proportion_test(100, 1000, 70, 1000, 0.05)
    assert r.difference == pytest.approx(-0.03) and r.relative_change == pytest.approx(-0.3)
    assert r.ci_low < r.difference < r.ci_high and r.p_value < 0.05
    same = xs.two_proportion_test(100, 1000, 100, 1000, 0.05)
    assert same.p_value == pytest.approx(1.0)


def test_srm_check_flags_a_broken_split() -> None:
    assert not xs.srm_check(2000, 8000, 0.8, 0.001)["mismatch"]
    assert xs.srm_check(1700, 8000, 0.8, 0.001)["mismatch"]


def test_assignment_is_exact_and_seeded() -> None:
    a = simulate.assign(1000, 0.8, np.random.default_rng(1))
    b = simulate.assign(1000, 0.8, np.random.default_rng(1))
    assert a.sum() == 800 and (a == b).all()


def test_simulated_effect_reduces_churn_on_average() -> None:
    rng = np.random.default_rng(0)
    p = np.full(200_000, 0.3)
    treated = simulate.assign(len(p), 0.5, rng)
    y = simulate.simulate_outcomes(p, treated, np.full(len(p), 0.2), rng)
    assert y[~treated].mean() == pytest.approx(0.30, abs=0.005)
    assert y[treated].mean() == pytest.approx(0.24, abs=0.005)
