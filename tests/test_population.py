import itertools

import numpy as np
import pytest

from paired_auc import (
    UndefinedPopulationFunctionalError,
    empirical_plugin_bounds,
    paired_bounds_exact_count,
    population_bounds_from_components,
    population_bounds_prevalence_set,
    trimmed_expectation_bounds,
)


TOL = 1e-12


def test_trimmed_expectation_fractional_boundary_mass():
    lower, upper = trimmed_expectation_bounds([-2.0, 0.0, 4.0], 0.5, [0.2, 0.5, 0.3])
    assert lower == pytest.approx(-0.4, abs=TOL)
    assert upper == pytest.approx(1.2, abs=TOL)


@pytest.mark.parametrize("mass", [0.0, 1.0])
def test_trimmed_expectation_boundary_mass(mass):
    values = np.array([-1.0, 0.5, 1.0])
    lower, upper = trimmed_expectation_bounds(values, mass)
    expected = 0.0 if mass == 0.0 else float(np.mean(values))
    assert lower == pytest.approx(expected, abs=TOL)
    assert upper == pytest.approx(expected, abs=TOL)


def test_population_component_formula():
    result = population_bounds_from_components(
        rho=0.5,
        p1=0.1,
        mu1=0.02,
        unverified_d_values=[-0.4, 0.0, 0.6],
        prevalence=0.3,
    )
    assert result.tau == pytest.approx(0.4, abs=TOL)
    lower_tail, upper_tail = trimmed_expectation_bounds([-0.4, 0.0, 0.6], 0.4)
    assert result.lower == pytest.approx((0.02 + 0.5 * lower_tail) / 0.21, abs=TOL)
    assert result.upper == pytest.approx((0.02 + 0.5 * upper_tail) / 0.21, abs=TOL)


def test_population_prevalence_set_is_endpoint_hull():
    low, high, results = population_bounds_prevalence_set(
        0.5, 0.1, 0.02, [-0.4, 0.0, 0.6], [0.2, 0.3, 0.4]
    )
    assert low == min(item.lower for item in results)
    assert high == max(item.upper for item in results)


def test_exact_finite_sample_plugin_identity_with_ties():
    a = np.array([1, 1, 3, 5, 4, 2], dtype=float)
    b = np.array([4, 2, 2, 1, 5, 3], dtype=float)
    labels = np.array([1, np.nan, 0, np.nan, np.nan, 1])
    verified = np.array([1, 0, 1, 0, 0, 1], dtype=bool)
    m = 3
    finite = paired_bounds_exact_count(a, b, labels, verified, m)
    plugin = empirical_plugin_bounds(a, b, labels, verified, m / float(a.size))
    assert not plugin.projection_used
    assert plugin.lower == pytest.approx(finite.lower, abs=TOL, rel=TOL)
    assert plugin.upper == pytest.approx(finite.upper, abs=TOL, rel=TOL)


def test_plugin_projection_is_explicit():
    result = empirical_plugin_bounds(
        [1, 2, 3, 4], [4, 3, 2, 1], [1, 1, np.nan, np.nan], [1, 1, 0, 0], 0.25
    )
    assert result.projection_used
    assert result.tau_hat == 0.0


def test_plugin_undefined_without_unverified_subjects():
    with pytest.raises(UndefinedPopulationFunctionalError):
        empirical_plugin_bounds([1, 2, 3], [3, 2, 1], [1, 0, 0], [1, 1, 1], 1.0 / 3.0)


def test_discrete_tail_allocation_matches_grid_vertices():
    values = np.array([-1.0, 0.5, 2.0])
    probabilities = np.array([0.2, 0.3, 0.5])
    tau = 0.4
    lower, upper = trimmed_expectation_bounds(values, tau, probabilities)
    candidates = []
    for fractional_index in range(3):
        others = [i for i in range(3) if i != fractional_index]
        for bits in itertools.product([0.0, 1.0], repeat=2):
            g = np.zeros(3)
            g[others] = bits
            g[fractional_index] = (tau - np.dot(probabilities, g)) / probabilities[fractional_index]
            if -TOL <= g[fractional_index] <= 1.0 + TOL:
                candidates.append(float(np.dot(probabilities * values, g)))
    assert lower == pytest.approx(min(candidates), abs=TOL)
    assert upper == pytest.approx(max(candidates), abs=TOL)
