"""Finite-cohort sharp bounds and the separate-AUC comparator."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .core import (
    EmptyCountSetError,
    IncompatibleCountError,
    InputValidationError,
    UndefinedAUCError,
    ascending_midranks,
    paired_rank_contrasts,
    validate_observed_labels,
    validate_score_pair,
    validate_scores,
)


@dataclass(frozen=True)
class ExactBounds:
    lower: float
    upper: float
    total_positive_count: int
    unverified_positive_count: int
    verified_positive_count: int
    verified_negative_count: int
    unverified_count: int

    @property
    def width(self):
        return self.upper - self.lower


@dataclass(frozen=True)
class CountSetBounds:
    lower: float
    upper: float
    counts: tuple
    lower_attaining_counts: tuple
    upper_attaining_counts: tuple
    per_count: tuple

    @property
    def width(self):
        return self.upper - self.lower


@dataclass(frozen=True)
class SeparateBounds:
    lower: float
    upper: float
    auc_a_lower: float
    auc_a_upper: float
    auc_b_lower: float
    auc_b_upper: float
    counts: tuple

    @property
    def width(self):
        return self.upper - self.lower


@dataclass(frozen=True)
class IntervalComparison:
    contained: bool
    lower_margin: float
    upper_margin: float


def _prepare(scores_a, scores_b, labels, verified):
    a, b = validate_score_pair(scores_a, scores_b)
    y, r = validate_observed_labels(labels, verified, a.size)
    d = paired_rank_contrasts(a, b)
    p = int(np.sum(y[r]))
    v = int(np.sum(r))
    u = int(a.size - v)
    z = int(v - p)
    c_v = float(np.dot(y[r], d[r]))
    d_unverified = np.sort(d[~r])
    prefix = np.concatenate(([0.0], np.cumsum(d_unverified, dtype=float)))
    return a, b, y, r, d, p, z, u, c_v, prefix


def _coerce_count(m):
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
        raise InputValidationError("total positive count must be an integer")
    return int(m)


def _validate_count(m, n, p, u):
    m = _coerce_count(m)
    if m == 0 or m == n:
        raise UndefinedAUCError("AUC is undefined for total positive count {}".format(m))
    if m < 0 or m > n or m < p or m > p + u:
        raise IncompatibleCountError(
            "total positive count {} is incompatible with p={} and u={}".format(m, p, u)
        )
    return m, m - p


def _exact_from_prefix(m, n, p, z, u, c_v, prefix):
    m, q = _validate_count(m, n, p, u)
    denominator = float(m * (n - m))
    lower = (c_v + prefix[q]) / denominator
    upper = (c_v + prefix[u] - prefix[u - q]) / denominator
    return ExactBounds(lower, upper, m, q, p, z, u)


def _normalize_count_set(counts):
    try:
        values = list(counts)
    except TypeError:
        raise InputValidationError("counts must be an iterable of integers")
    if not values:
        raise EmptyCountSetError("count set must not be empty")
    return tuple(sorted(set(_coerce_count(value) for value in values)))


def paired_bounds_exact_count(scores_a, scores_b, labels, verified, m):
    """Theorem 3.1 using sorting and prefix sums, never enumeration."""

    a, _, _, _, _, p, z, u, c_v, prefix = _prepare(
        scores_a, scores_b, labels, verified
    )
    return _exact_from_prefix(m, a.size, p, z, u, c_v, prefix)


def paired_bounds_count_set(scores_a, scores_b, labels, verified, counts):
    """Corollary 3.3: envelope over a nonempty set of feasible counts."""

    normalized = _normalize_count_set(counts)
    a, _, _, _, _, p, z, u, c_v, prefix = _prepare(
        scores_a, scores_b, labels, verified
    )
    per_count = tuple(
        _exact_from_prefix(m, a.size, p, z, u, c_v, prefix) for m in normalized
    )
    lower = min(item.lower for item in per_count)
    upper = max(item.upper for item in per_count)
    tol = 1e-14
    lower_counts = tuple(item.total_positive_count for item in per_count if abs(item.lower - lower) <= tol)
    upper_counts = tuple(item.total_positive_count for item in per_count if abs(item.upper - upper) <= tol)
    return CountSetBounds(lower, upper, normalized, lower_counts, upper_counts, per_count)


def compatible_total_positive_counts(labels, verified, n=None):
    if n is None:
        n = len(labels)
    y, r = validate_observed_labels(labels, verified, n)
    p = int(np.sum(y[r]))
    u = int(n - np.sum(r))
    lower = max(1, p)
    upper = min(n - 1, p + u)
    if lower > upper:
        raise IncompatibleCountError("no compatible total count yields both outcome classes")
    return tuple(range(lower, upper + 1))


def paired_bounds_unrestricted(scores_a, scores_b, labels, verified):
    a, _ = validate_score_pair(scores_a, scores_b)
    counts = compatible_total_positive_counts(labels, verified, a.size)
    return paired_bounds_count_set(scores_a, scores_b, labels, verified, counts)


def _single_auc_exact_from_ranks(ranks, y, r, m):
    n = ranks.size
    p = int(np.sum(y[r]))
    u = int(n - np.sum(r))
    m, q = _validate_count(m, n, p, u)
    verified_rank_sum = float(np.dot(y[r], ranks[r]))
    unverified = np.sort(ranks[~r])
    prefix = np.concatenate(([0.0], np.cumsum(unverified, dtype=float)))
    correction = m * (m + 1) / 2.0
    denominator = float(m * (n - m))
    lower = (verified_rank_sum + prefix[q] - correction) / denominator
    upper = (verified_rank_sum + prefix[u] - prefix[u - q] - correction) / denominator
    return lower, upper


def auc_bounds_exact_count(scores, labels, verified, m):
    s = validate_scores(scores)
    if s.size < 2:
        raise InputValidationError("at least two subjects are required")
    y, r = validate_observed_labels(labels, verified, s.size)
    return _single_auc_exact_from_ranks(ascending_midranks(s), y, r, m)


def _auc_bounds_count_set_from_ranks(ranks, y, r, counts):
    values = tuple(_single_auc_exact_from_ranks(ranks, y, r, m) for m in counts)
    return min(value[0] for value in values), max(value[1] for value in values)


def separate_difference_bounds_exact_count(scores_a, scores_b, labels, verified, m):
    a, b = validate_score_pair(scores_a, scores_b)
    y, r = validate_observed_labels(labels, verified, a.size)
    m = _coerce_count(m)
    a_lower, a_upper = _single_auc_exact_from_ranks(ascending_midranks(a), y, r, m)
    b_lower, b_upper = _single_auc_exact_from_ranks(ascending_midranks(b), y, r, m)
    return SeparateBounds(
        a_lower - b_upper,
        a_upper - b_lower,
        a_lower,
        a_upper,
        b_lower,
        b_upper,
        (m,),
    )


def separate_difference_bounds_count_set(scores_a, scores_b, labels, verified, counts):
    normalized = _normalize_count_set(counts)
    a, b = validate_score_pair(scores_a, scores_b)
    y, r = validate_observed_labels(labels, verified, a.size)
    # Validate all counts before forming either model's envelope.
    p = int(np.sum(y[r]))
    u = int(a.size - np.sum(r))
    for m in normalized:
        _validate_count(m, a.size, p, u)
    a_lower, a_upper = _auc_bounds_count_set_from_ranks(ascending_midranks(a), y, r, normalized)
    b_lower, b_upper = _auc_bounds_count_set_from_ranks(ascending_midranks(b), y, r, normalized)
    return SeparateBounds(
        a_lower - b_upper,
        a_upper - b_lower,
        a_lower,
        a_upper,
        b_lower,
        b_upper,
        normalized,
    )


def separate_difference_bounds_unrestricted(scores_a, scores_b, labels, verified):
    a, _ = validate_score_pair(scores_a, scores_b)
    counts = compatible_total_positive_counts(labels, verified, a.size)
    return separate_difference_bounds_count_set(scores_a, scores_b, labels, verified, counts)


def compare_paired_separate(paired, separate, tolerance=1e-12):
    if tolerance < 0 or not np.isfinite(tolerance):
        raise InputValidationError("tolerance must be finite and nonnegative")
    lower_margin = paired.lower - separate.lower
    upper_margin = separate.upper - paired.upper
    return IntervalComparison(
        lower_margin >= -tolerance and upper_margin >= -tolerance,
        lower_margin,
        upper_margin,
    )


def paired_bounds_stratified_counts(
    scores_a, scores_b, labels, verified, strata, counts_by_stratum
):
    """Prose extension in Section 3.5: exact positive count per stratum."""

    a, b = validate_score_pair(scores_a, scores_b)
    y, r = validate_observed_labels(labels, verified, a.size)
    strata_array = np.asarray(strata)
    if strata_array.ndim != 1 or strata_array.size != a.size:
        raise InputValidationError("strata must be one-dimensional and subject aligned")
    d = paired_rank_contrasts(a, b)
    c_v = float(np.dot(y[r], d[r]))
    lower_sum = 0.0
    upper_sum = 0.0
    total_m = 0
    unique_strata = list(dict.fromkeys(strata_array.tolist()))
    if set(unique_strata) != set(counts_by_stratum.keys()):
        raise InputValidationError("counts_by_stratum must contain exactly the observed strata")
    for stratum in unique_strata:
        in_h = strata_array == stratum
        p_h = int(np.sum(y[r & in_h]))
        u_h = int(np.sum((~r) & in_h))
        m_h = _coerce_count(counts_by_stratum[stratum])
        q_h = m_h - p_h
        if q_h < 0 or q_h > u_h:
            raise IncompatibleCountError("incompatible positive count in stratum {!r}".format(stratum))
        ordered = np.sort(d[(~r) & in_h])
        prefix = np.concatenate(([0.0], np.cumsum(ordered, dtype=float)))
        lower_sum += prefix[q_h]
        upper_sum += prefix[u_h] - prefix[u_h - q_h]
        total_m += m_h
    if total_m == 0 or total_m == a.size:
        raise UndefinedAUCError("stratified total count leaves one outcome class empty")
    denominator = float(total_m * (a.size - total_m))
    return ExactBounds(
        (c_v + lower_sum) / denominator,
        (c_v + upper_sum) / denominator,
        total_m,
        total_m - int(np.sum(y[r])),
        int(np.sum(y[r])),
        int(np.sum(r) - np.sum(y[r])),
        int(a.size - np.sum(r)),
    )


def auc_contrast_bounds_exact_count(scores, coefficients, labels, verified, m):
    """Corollary 3.6 for a zero-sum contrast of K empirical AUCs."""

    matrix = np.asarray(scores, dtype=float)
    coeff = np.asarray(coefficients, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise InputValidationError("scores must have shape (n, K) with n,K >= 2")
    if coeff.ndim != 1 or coeff.size != matrix.shape[1] or not np.all(np.isfinite(coeff)):
        raise InputValidationError("coefficients must be a finite length-K vector")
    if not np.all(np.isfinite(matrix)):
        raise InputValidationError("scores must be finite")
    if abs(float(np.sum(coeff))) > 1e-12:
        raise InputValidationError("coefficients must sum to zero")
    y, r = validate_observed_labels(labels, verified, matrix.shape[0])
    ranks = np.column_stack([ascending_midranks(matrix[:, k]) for k in range(matrix.shape[1])])
    objective = np.dot(ranks, coeff)
    p = int(np.sum(y[r]))
    u = int(matrix.shape[0] - np.sum(r))
    z = int(np.sum(r) - p)
    c_v = float(np.dot(y[r], objective[r]))
    ordered = np.sort(objective[~r])
    prefix = np.concatenate(([0.0], np.cumsum(ordered, dtype=float)))
    return _exact_from_prefix(m, matrix.shape[0], p, z, u, c_v, prefix)
