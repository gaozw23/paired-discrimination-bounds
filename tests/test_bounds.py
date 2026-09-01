import numpy as np
import pytest

from paired_auc import (
    EmptyCountSetError,
    IncompatibleCountError,
    InputValidationError,
    UndefinedAUCError,
    auc_contrast_bounds_exact_count,
    compare_paired_separate,
    paired_auc_difference,
    paired_bounds_count_set,
    paired_bounds_exact_count,
    paired_bounds_stratified_counts,
    paired_bounds_unrestricted,
    separate_difference_bounds_count_set,
    separate_difference_bounds_exact_count,
)
from tests.brute_force_oracle import brute_force_count_set, brute_force_exact


TOL = 1e-12


def toy_data():
    # The scores are chosen to realize exactly the manuscript's displayed ranks.
    rank_a = np.array([7, 1, 2, 5, 3, 6, 4, 8], dtype=float)
    rank_b = np.array([7, 1, 3, 6, 8, 4, 2, 5], dtype=float)
    labels = np.array([1, np.nan, 0, np.nan, 0, 1, np.nan, np.nan])
    verified = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=bool)
    return rank_a, rank_b, labels, verified


def test_manuscript_toy_example_exactly():
    a, b, labels, verified = toy_data()
    result = paired_bounds_exact_count(a, b, labels, verified, 3)
    separate = separate_difference_bounds_exact_count(a, b, labels, verified, 3)
    assert result.lower == pytest.approx(1.0 / 15.0, abs=TOL)
    assert result.upper == pytest.approx(5.0 / 15.0, abs=TOL)
    assert separate.lower == pytest.approx(-3.0 / 15.0, abs=TOL)
    assert separate.upper == pytest.approx(9.0 / 15.0, abs=TOL)


def test_complete_labels_collapse_to_observed_difference():
    a = [3, 1, 4, 2, 5]
    b = [1, 4, 2, 3, 5]
    y = np.array([1, 0, 1, 0, 0], dtype=float)
    r = np.ones(5, dtype=bool)
    result = paired_bounds_exact_count(a, b, y, r, 2)
    expected = paired_auc_difference(a, b, y.astype(int))
    assert result.lower == pytest.approx(expected, abs=TOL)
    assert result.upper == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize("m", [1, 4])
def test_q_boundary_unique_completion(m):
    # One verified positive, one verified negative, and three unverified labels.
    # Thus m=1 gives q=0 and m=4 gives q=u while both classes remain present.
    a = [4, 1, 3, 2, 5]
    b = [1, 4, 2, 5, 3]
    labels = [1, np.nan, np.nan, np.nan, 0]
    verified = [1, 0, 0, 0, 1]
    result = paired_bounds_exact_count(a, b, labels, verified, m)
    assert result.lower == pytest.approx(result.upper, abs=TOL)


def test_all_labels_unverified_with_legal_count():
    result = paired_bounds_exact_count(
        [1, 4, 2, 3], [4, 1, 3, 2], [np.nan] * 4, [0] * 4, 2
    )
    oracle = brute_force_exact(
        [1, 4, 2, 3], [4, 1, 3, 2], [np.nan] * 4, [0] * 4, 2
    )
    assert result.lower == pytest.approx(oracle["paired_lower"], abs=TOL)
    assert result.upper == pytest.approx(oracle["paired_upper"], abs=TOL)


def test_exact_and_set_bounds_match_bruteforce():
    a, b, labels, verified = toy_data()
    exact = paired_bounds_exact_count(a, b, labels, verified, 3)
    oracle_exact = brute_force_exact(a, b, labels, verified, 3)
    assert exact.lower == pytest.approx(oracle_exact["paired_lower"], abs=TOL)
    assert exact.upper == pytest.approx(oracle_exact["paired_upper"], abs=TOL)
    counts = [4, 2, 3, 3]
    envelope = paired_bounds_count_set(a, b, labels, verified, counts)
    oracle_set = brute_force_count_set(a, b, labels, verified, counts)
    assert envelope.counts == (2, 3, 4)
    assert envelope.lower == pytest.approx(oracle_set["paired_lower"], abs=TOL)
    assert envelope.upper == pytest.approx(oracle_set["paired_upper"], abs=TOL)


def test_unrestricted_matches_bruteforce():
    a, b, labels, verified = toy_data()
    result = paired_bounds_unrestricted(a, b, labels, verified)
    oracle = brute_force_count_set(a, b, labels, verified, result.counts)
    assert result.lower == pytest.approx(oracle["paired_lower"], abs=TOL)
    assert result.upper == pytest.approx(oracle["paired_upper"], abs=TOL)


def test_paired_is_contained_in_separate_exact_and_set():
    a, b, labels, verified = toy_data()
    paired = paired_bounds_exact_count(a, b, labels, verified, 3)
    separate = separate_difference_bounds_exact_count(a, b, labels, verified, 3)
    assert compare_paired_separate(paired, separate).contained
    paired_set = paired_bounds_count_set(a, b, labels, verified, [2, 3, 4])
    separate_set = separate_difference_bounds_count_set(a, b, labels, verified, [2, 3, 4])
    assert compare_paired_separate(paired_set, separate_set).contained


def test_model_swap_symmetry():
    a, b, labels, verified = toy_data()
    original = paired_bounds_count_set(a, b, labels, verified, [2, 3, 4])
    swapped = paired_bounds_count_set(b, a, labels, verified, [2, 3, 4])
    assert swapped.lower == pytest.approx(-original.upper, abs=TOL)
    assert swapped.upper == pytest.approx(-original.lower, abs=TOL)


def test_subject_permutation_invariance():
    a, b, labels, verified = toy_data()
    order = np.array([5, 1, 7, 0, 3, 6, 2, 4])
    original = paired_bounds_count_set(a, b, labels, verified, [2, 3, 4])
    permuted = paired_bounds_count_set(a[order], b[order], labels[order], verified[order], [2, 3, 4])
    assert permuted.lower == pytest.approx(original.lower, abs=TOL)
    assert permuted.upper == pytest.approx(original.upper, abs=TOL)


def test_strict_monotone_transform_invariance():
    a, b, labels, verified = toy_data()
    original = paired_bounds_count_set(a, b, labels, verified, [2, 3, 4])
    transformed = paired_bounds_count_set(np.exp(a), 3.0 * b + 7.0, labels, verified, [2, 3, 4])
    assert transformed.lower == pytest.approx(original.lower, abs=TOL)
    assert transformed.upper == pytest.approx(original.upper, abs=TOL)


def test_rank_contrast_ties_do_not_change_endpoint_value():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 5, 4, 3]
    labels = [1, np.nan, np.nan, np.nan, 0]
    verified = [1, 0, 0, 0, 1]
    result = paired_bounds_exact_count(a, b, labels, verified, 2)
    oracle = brute_force_exact(a, b, labels, verified, 2)
    assert result.lower == pytest.approx(oracle["paired_lower"], abs=TOL)
    assert result.upper == pytest.approx(oracle["paired_upper"], abs=TOL)


@pytest.mark.parametrize("m", [-1, 1, 6])
def test_incompatible_count_rejected(m):
    with pytest.raises(IncompatibleCountError):
        paired_bounds_exact_count([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 1, np.nan, 0, np.nan], [1, 1, 0, 1, 0], m)


def test_empty_count_set_rejected():
    with pytest.raises(EmptyCountSetError):
        paired_bounds_count_set([1, 2], [2, 1], [np.nan, np.nan], [0, 0], [])


@pytest.mark.parametrize("m", [0, 5])
def test_single_class_count_is_undefined(m):
    with pytest.raises(UndefinedAUCError):
        paired_bounds_exact_count([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [np.nan] * 5, [0] * 5, m)


def test_count_set_with_any_illegal_value_fails_strictly():
    with pytest.raises(UndefinedAUCError):
        paired_bounds_count_set([1, 2, 3], [3, 2, 1], [np.nan] * 3, [0] * 3, [0, 1, 2])
    with pytest.raises(InputValidationError):
        paired_bounds_count_set([1, 2, 3], [3, 2, 1], [np.nan] * 3, [0] * 3, [1, 1.5])


def test_stratified_extension_matches_manual_bruteforce_filter():
    a = np.array([1, 4, 2, 5, 3, 6], dtype=float)
    b = np.array([6, 1, 4, 2, 5, 3], dtype=float)
    labels = np.array([1, np.nan, 0, np.nan, np.nan, np.nan])
    verified = np.array([1, 0, 1, 0, 0, 0], dtype=bool)
    strata = np.array([0, 0, 0, 1, 1, 1])
    result = paired_bounds_stratified_counts(a, b, labels, verified, strata, {0: 1, 1: 2})
    oracle = brute_force_exact(a, b, labels, verified, 3)
    values = []
    for completed, _, _, delta in oracle["rows"]:
        completed = np.asarray(completed)
        if int(np.sum(completed[strata == 0])) == 1 and int(np.sum(completed[strata == 1])) == 2:
            values.append(delta)
    assert result.lower == pytest.approx(min(values), abs=TOL)
    assert result.upper == pytest.approx(max(values), abs=TOL)


def test_zero_sum_auc_contrast_reproduces_paired_bounds():
    a, b, labels, verified = toy_data()
    paired = paired_bounds_exact_count(a, b, labels, verified, 3)
    multiple = auc_contrast_bounds_exact_count(np.column_stack([a, b]), [1.0, -1.0], labels, verified, 3)
    assert multiple.lower == pytest.approx(paired.lower, abs=TOL)
    assert multiple.upper == pytest.approx(paired.upper, abs=TOL)
    with pytest.raises(InputValidationError):
        auc_contrast_bounds_exact_count(np.column_stack([a, b]), [1.0, 0.0], labels, verified, 3)
