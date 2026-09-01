import numpy as np
import pytest

from paired_auc import (
    compare_paired_separate,
    paired_bounds_count_set,
    paired_bounds_exact_count,
    paired_bounds_unrestricted,
    separate_difference_bounds_count_set,
    separate_difference_bounds_exact_count,
)
from tests.brute_force_oracle import brute_force_count_set, brute_force_exact


TOL = 1e-12


def test_fixed_seed_random_small_cases_against_oracle():
    rng = np.random.RandomState(20260901)
    checked = 0
    while checked < 240:
        n = int(rng.randint(4, 11))
        if checked % 2 == 0:
            a = rng.randint(-2, 4, size=n).astype(float)
            b = rng.randint(-2, 4, size=n).astype(float)
        else:
            a = rng.normal(size=n)
            b = rng.normal(size=n)
        full_y = rng.randint(0, 2, size=n)
        m = int(np.sum(full_y))
        if m == 0 or m == n:
            continue
        verified = rng.rand(n) < 0.55
        labels = np.full(n, np.nan)
        labels[verified] = full_y[verified]

        analytic = paired_bounds_exact_count(a, b, labels, verified, m)
        oracle = brute_force_exact(a, b, labels, verified, m)
        assert analytic.lower == pytest.approx(oracle["paired_lower"], abs=TOL, rel=TOL)
        assert analytic.upper == pytest.approx(oracle["paired_upper"], abs=TOL, rel=TOL)

        separate = separate_difference_bounds_exact_count(a, b, labels, verified, m)
        assert separate.auc_a_lower == pytest.approx(oracle["auc_a_lower"], abs=TOL, rel=TOL)
        assert separate.auc_a_upper == pytest.approx(oracle["auc_a_upper"], abs=TOL, rel=TOL)
        assert separate.auc_b_lower == pytest.approx(oracle["auc_b_lower"], abs=TOL, rel=TOL)
        assert separate.auc_b_upper == pytest.approx(oracle["auc_b_upper"], abs=TOL, rel=TOL)
        assert compare_paired_separate(analytic, separate, TOL).contained

        p = int(np.sum(labels[verified]))
        u = int(np.sum(~verified))
        feasible = list(range(max(1, p), min(n - 1, p + u) + 1))
        chosen_counts = feasible[::2] or feasible[:1]
        analytic_set = paired_bounds_count_set(a, b, labels, verified, chosen_counts)
        oracle_set = brute_force_count_set(a, b, labels, verified, chosen_counts)
        assert analytic_set.lower == pytest.approx(oracle_set["paired_lower"], abs=TOL, rel=TOL)
        assert analytic_set.upper == pytest.approx(oracle_set["paired_upper"], abs=TOL, rel=TOL)
        separate_set = separate_difference_bounds_count_set(a, b, labels, verified, chosen_counts)
        assert compare_paired_separate(analytic_set, separate_set, TOL).contained

        unrestricted = paired_bounds_unrestricted(a, b, labels, verified)
        oracle_unrestricted = brute_force_count_set(a, b, labels, verified, unrestricted.counts)
        assert unrestricted.lower == pytest.approx(oracle_unrestricted["paired_lower"], abs=TOL, rel=TOL)
        assert unrestricted.upper == pytest.approx(oracle_unrestricted["paired_upper"], abs=TOL, rel=TOL)

        order = rng.permutation(n)
        permuted = paired_bounds_exact_count(a[order], b[order], labels[order], verified[order], m)
        assert permuted.lower == pytest.approx(analytic.lower, abs=TOL, rel=TOL)
        assert permuted.upper == pytest.approx(analytic.upper, abs=TOL, rel=TOL)

        transformed = paired_bounds_exact_count(2.0 * a + 3.0, np.exp(b), labels, verified, m)
        assert transformed.lower == pytest.approx(analytic.lower, abs=TOL, rel=TOL)
        assert transformed.upper == pytest.approx(analytic.upper, abs=TOL, rel=TOL)

        swapped = paired_bounds_exact_count(b, a, labels, verified, m)
        assert swapped.lower == pytest.approx(-analytic.upper, abs=TOL, rel=TOL)
        assert swapped.upper == pytest.approx(-analytic.lower, abs=TOL, rel=TOL)
        checked += 1
