import numpy as np
import pytest

from paired_auc import (
    InputValidationError,
    UndefinedAUCError,
    ascending_midranks,
    complete_case_auc_difference,
    empirical_auc,
    empirical_auc_pairwise,
    paired_auc_difference,
    paired_rank_contrasts,
    tie_adjusted_comparison,
    verified_label_counts,
)


TOL = 1e-12


def test_tie_adjusted_comparison_definition():
    assert np.array_equal(
        tie_adjusted_comparison(np.array([-2.0, 0.0, 3.0])),
        np.array([0.0, 0.5, 1.0]),
    )


def test_ascending_midranks_with_ties():
    assert np.array_equal(
        ascending_midranks([4.0, 1.0, 1.0, 3.0]),
        np.array([4.0, 1.5, 1.5, 3.0]),
    )


def test_all_tied_midranks():
    assert np.array_equal(ascending_midranks([2, 2, 2]), np.array([2.0, 2.0, 2.0]))


@pytest.mark.parametrize(
    "scores,labels",
    [
        ([0, 1, 2, 3], [0, 0, 1, 1]),
        ([0, 0, 2, 2], [0, 1, 0, 1]),
        ([3, 1, 3, 2, 1], [1, 0, 0, 1, 0]),
    ],
)
def test_pairwise_auc_equals_rank_auc(scores, labels):
    assert empirical_auc(scores, labels) == pytest.approx(
        empirical_auc_pairwise(scores, labels), abs=TOL, rel=TOL
    )


def test_paired_rank_identity_matches_auc_difference():
    a = np.array([3, 1, 4, 2, 2], dtype=float)
    b = np.array([1, 4, 2, 2, 3], dtype=float)
    y = np.array([1, 0, 1, 0, 0])
    expected = empirical_auc(a, y) - empirical_auc(b, y)
    assert paired_auc_difference(a, b, y) == pytest.approx(expected, abs=TOL, rel=TOL)


def test_paired_rank_contrast_sums_to_zero():
    d = paired_rank_contrasts([1, 1, 3, 4], [4, 2, 2, 1])
    assert float(np.sum(d)) == pytest.approx(0.0, abs=TOL)


def test_verified_counts_allow_nan_only_unverified():
    result = verified_label_counts([1, np.nan, 0, np.nan], [1, 0, 1, 0])
    assert result.verified_positive_count == 1
    assert result.verified_negative_count == 1
    assert result.unverified_count == 2


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1]])
def test_auc_undefined_without_both_classes(labels):
    with pytest.raises(UndefinedAUCError):
        empirical_auc([1, 2, 3], labels)
    with pytest.raises(UndefinedAUCError):
        empirical_auc_pairwise([1, 2, 3], labels)


def test_complete_case_undefined_without_verified_class():
    with pytest.raises(UndefinedAUCError):
        complete_case_auc_difference([1, 2, 3], [3, 2, 1], [1, np.nan, np.nan], [1, 0, 0])


@pytest.mark.parametrize("bad", [[1.0, np.nan], [1.0, np.inf], [1.0, -np.inf]])
def test_nonfinite_scores_rejected(bad):
    with pytest.raises(InputValidationError):
        ascending_midranks(bad)


def test_illegal_labels_rejected():
    with pytest.raises(InputValidationError):
        empirical_auc([1, 2, 3], [0, 2, 1])
    with pytest.raises(InputValidationError):
        verified_label_counts([0, 2, np.nan], [1, 1, 0])
    with pytest.raises(InputValidationError):
        verified_label_counts([0, np.inf, np.nan], [1, 0, 0])
    with pytest.raises(InputValidationError):
        verified_label_counts([0, 2, np.nan], [1, 0, 0])


def test_different_lengths_rejected():
    with pytest.raises(InputValidationError):
        paired_rank_contrasts([1, 2], [1, 2, 3])
    with pytest.raises(InputValidationError):
        empirical_auc([1, 2, 3], [0, 1])
