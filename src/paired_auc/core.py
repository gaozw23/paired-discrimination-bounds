"""Core tie-aware rank and empirical AUC calculations.

The definitions follow equations (1)--(6) of the canonical manuscript.
"""

from dataclasses import dataclass

import numpy as np


class PairedAUCError(ValueError):
    """Base class for explicit mathematical/input failures."""


class InputValidationError(PairedAUCError):
    """The supplied arrays or values do not satisfy the API contract."""


class UndefinedAUCError(PairedAUCError):
    """An AUC is undefined because one outcome class is absent."""


class IncompatibleCountError(PairedAUCError):
    """A proposed total count is incompatible with the observed labels."""


class EmptyCountSetError(PairedAUCError):
    """No total-positive count was supplied."""


@dataclass(frozen=True)
class VerifiedLabelCounts:
    n: int
    verified_count: int
    unverified_count: int
    verified_positive_count: int
    verified_negative_count: int


def _as_1d_array(values, name, dtype=float):
    try:
        array = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise InputValidationError("{} cannot be converted to the required numeric type".format(name)) from error
    if array.ndim != 1:
        raise InputValidationError("{} must be a one-dimensional array".format(name))
    if array.size == 0:
        raise InputValidationError("{} must not be empty".format(name))
    return array


def validate_scores(scores, name="scores"):
    """Return a finite one-dimensional float score vector."""

    array = _as_1d_array(scores, name, dtype=float)
    if not np.all(np.isfinite(array)):
        raise InputValidationError("{} must contain only finite values".format(name))
    return array


def validate_score_pair(scores_a, scores_b):
    a = validate_scores(scores_a, "scores_a")
    b = validate_scores(scores_b, "scores_b")
    if a.size != b.size:
        raise InputValidationError("scores_a and scores_b must have equal length")
    if a.size < 2:
        raise InputValidationError("at least two subjects are required")
    return a, b


def validate_complete_labels(labels, n):
    y = _as_1d_array(labels, "labels", dtype=float)
    if y.size != n:
        raise InputValidationError("labels must have the same length as scores")
    if not np.all(np.isfinite(y)) or not np.all((y == 0.0) | (y == 1.0)):
        raise InputValidationError("complete labels must contain only 0 and 1")
    return y.astype(np.int8)


def validate_observed_labels(labels, verified, n):
    """Validate labels on verified subjects and ignore placeholders elsewhere."""

    y = _as_1d_array(labels, "labels", dtype=float)
    r_raw = np.asarray(verified)
    if r_raw.ndim != 1 or r_raw.size == 0:
        raise InputValidationError("verified must be a nonempty one-dimensional array")
    if y.size != n or r_raw.size != n:
        raise InputValidationError("scores, labels, and verified must have equal length")
    if np.any(np.isinf(y)):
        raise InputValidationError("labels must not contain infinite values")
    finite_labels = y[np.isfinite(y)]
    if not np.all((finite_labels == 0.0) | (finite_labels == 1.0)):
        raise InputValidationError("labels must be binary where supplied")
    if r_raw.dtype == np.bool_:
        r = r_raw.astype(bool, copy=False)
    else:
        try:
            r_num = r_raw.astype(float)
        except (TypeError, ValueError):
            raise InputValidationError("verified must contain only Boolean/0/1 values")
        if not np.all(np.isfinite(r_num)) or not np.all((r_num == 0.0) | (r_num == 1.0)):
            raise InputValidationError("verified must contain only Boolean/0/1 values")
        r = r_num.astype(bool)
    observed = y[r]
    if not np.all(np.isfinite(observed)) or not np.all((observed == 0.0) | (observed == 1.0)):
        raise InputValidationError("verified labels must contain only 0 and 1")
    return y, r


def tie_adjusted_comparison(x):
    """Equation (1): 1(x>0) + 0.5*1(x=0)."""

    array = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(array)):
        raise InputValidationError("comparison input must be finite")
    result = (array > 0.0).astype(float) + 0.5 * (array == 0.0).astype(float)
    if result.ndim == 0:
        return float(result)
    return result


def ascending_midranks(scores):
    """Equation (3), implemented in O(n log n) time by tied blocks."""

    x = validate_scores(scores)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and x[order[end]] == x[order[start]]:
            end += 1
        # The block occupies one-based ranks start+1 through end.
        midrank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = midrank
        start = end
    return ranks


def empirical_auc_pairwise(scores, labels):
    """Direct O(n^2) implementation of equation (2)."""

    s = validate_scores(scores)
    y = validate_complete_labels(labels, s.size)
    m = int(np.sum(y))
    if m == 0 or m == s.size:
        raise UndefinedAUCError("empirical AUC requires at least one positive and one negative")
    positive = s[y == 1]
    negative = s[y == 0]
    differences = positive[:, None] - negative[None, :]
    return float(np.sum(tie_adjusted_comparison(differences)) / (m * (s.size - m)))


def empirical_auc(scores, labels):
    """Rank-sum implementation of empirical AUC, equation (5)."""

    s = validate_scores(scores)
    y = validate_complete_labels(labels, s.size)
    m = int(np.sum(y))
    if m == 0 or m == s.size:
        raise UndefinedAUCError("empirical AUC requires at least one positive and one negative")
    ranks = ascending_midranks(s)
    numerator = float(np.dot(y, ranks)) - m * (m + 1) / 2.0
    return numerator / (m * (s.size - m))


def paired_rank_contrasts(scores_a, scores_b):
    """Equation (4): subject-aligned differences of ascending midranks."""

    a, b = validate_score_pair(scores_a, scores_b)
    return ascending_midranks(a) - ascending_midranks(b)


def paired_auc_difference(scores_a, scores_b, labels):
    """Complete-data paired AUC difference using equation (6)."""

    a, b = validate_score_pair(scores_a, scores_b)
    y = validate_complete_labels(labels, a.size)
    m = int(np.sum(y))
    if m == 0 or m == a.size:
        raise UndefinedAUCError("paired AUC difference requires both outcome classes")
    d = paired_rank_contrasts(a, b)
    return float(np.dot(y, d) / (m * (a.size - m)))


def verified_label_counts(labels, verified, n=None):
    if n is None:
        n = len(labels)
    y, r = validate_observed_labels(labels, verified, n)
    p = int(np.sum(y[r]))
    v = int(np.sum(r))
    return VerifiedLabelCounts(
        n=int(n),
        verified_count=v,
        unverified_count=int(n - v),
        verified_positive_count=p,
        verified_negative_count=int(v - p),
    )


def complete_case_auc_difference(scores_a, scores_b, labels, verified):
    """Complete-case reference from the numerical-study prose."""

    a, b = validate_score_pair(scores_a, scores_b)
    y, r = validate_observed_labels(labels, verified, a.size)
    observed_y = y[r]
    if observed_y.size < 2 or np.all(observed_y == 0.0) or np.all(observed_y == 1.0):
        raise UndefinedAUCError(
            "complete-case AUC difference requires a verified positive and negative"
        )
    return paired_auc_difference(a[r], b[r], observed_y)
