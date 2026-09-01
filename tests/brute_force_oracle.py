"""Small-sample exhaustive oracle independent of the analytic bounds."""

import itertools

import numpy as np


def _omega(x):
    return float(x > 0.0) + 0.5 * float(x == 0.0)


def _direct_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    m = int(np.sum(labels))
    n = labels.size
    if m == 0 or m == n:
        raise ValueError("AUC undefined")
    total = 0.0
    for i in range(n):
        if labels[i] != 1:
            continue
        for k in range(n):
            if labels[k] == 0:
                total += _omega(scores[i] - scores[k])
    return total / (m * (n - m))


def enumerate_completions(labels, verified, total_positive_count):
    labels = np.asarray(labels, dtype=float)
    verified = np.asarray(verified, dtype=bool)
    n = labels.size
    m = int(total_positive_count)
    if m <= 0 or m >= n:
        raise ValueError("AUC undefined")
    observed_positive = int(np.sum(labels[verified]))
    unknown = np.flatnonzero(~verified)
    q = m - observed_positive
    if q < 0 or q > unknown.size:
        raise ValueError("incompatible total count")
    for chosen in itertools.combinations(unknown.tolist(), q):
        completed = np.zeros(n, dtype=np.int8)
        completed[verified] = labels[verified].astype(np.int8)
        if chosen:
            completed[list(chosen)] = 1
        yield completed


def brute_force_exact(scores_a, scores_b, labels, verified, total_positive_count):
    rows = []
    for completed in enumerate_completions(labels, verified, total_positive_count):
        auc_a = _direct_auc(scores_a, completed)
        auc_b = _direct_auc(scores_b, completed)
        rows.append((tuple(int(value) for value in completed), auc_a, auc_b, auc_a - auc_b))
    if not rows:
        raise ValueError("no compatible completion")
    differences = tuple(row[3] for row in rows)
    return {
        "rows": tuple(rows),
        "attainable_differences": differences,
        "paired_lower": min(differences),
        "paired_upper": max(differences),
        "auc_a_lower": min(row[1] for row in rows),
        "auc_a_upper": max(row[1] for row in rows),
        "auc_b_lower": min(row[2] for row in rows),
        "auc_b_upper": max(row[2] for row in rows),
    }


def brute_force_count_set(scores_a, scores_b, labels, verified, counts):
    results = tuple(
        (int(m), brute_force_exact(scores_a, scores_b, labels, verified, int(m)))
        for m in sorted(set(counts))
    )
    if not results:
        raise ValueError("empty count set")
    all_rows = tuple(row for _, result in results for row in result["rows"])
    differences = tuple(row[3] for row in all_rows)
    return {
        "per_count": results,
        "rows": all_rows,
        "attainable_differences": differences,
        "paired_lower": min(differences),
        "paired_upper": max(differences),
        "auc_a_lower": min(row[1] for row in all_rows),
        "auc_a_upper": max(row[1] for row in all_rows),
        "auc_b_lower": min(row[2] for row in all_rows),
        "auc_b_upper": max(row[2] for row in all_rows),
    }
