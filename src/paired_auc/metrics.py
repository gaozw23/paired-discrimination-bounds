"""Metric definitions and numerical checks for simulation records."""

from __future__ import annotations

from typing import Any

import numpy as np


class SimulationValidationError(RuntimeError):
    """A numerical or implementation check was violated."""

    def __init__(self, message: str, record: dict[str, Any] | None = None):
        super().__init__(message)
        self.record = record or {}


def interval_metrics(
    paired: Any,
    separate: Any,
    delta_full_n: float,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Calculate widths, certification, and hard gates for one count regime."""

    paired_width = float(paired.upper - paired.lower)
    separate_width = float(separate.upper - separate.lower)
    degenerate_separate = separate_width <= tolerance
    if degenerate_separate:
        width_ratio = np.nan
        percentage_reduction = np.nan
    else:
        width_ratio = paired_width / separate_width
        percentage_reduction = 100.0 * (1.0 - width_ratio)

    positive = bool(paired.lower > tolerance)
    negative = bool(paired.upper < -tolerance)
    unresolved = bool(not positive and not negative)
    finite_truth_contained = bool(
        paired.lower - tolerance <= delta_full_n <= paired.upper + tolerance
    )
    paired_within_separate = bool(
        paired.lower >= separate.lower - tolerance
        and paired.upper <= separate.upper + tolerance
    )
    lower_le_upper = bool(
        paired.lower <= paired.upper + tolerance
        and separate.lower <= separate.upper + tolerance
    )

    values = np.asarray(
        [
            paired.lower,
            paired.upper,
            paired_width,
            separate.lower,
            separate.upper,
            separate_width,
        ],
        dtype=float,
    )
    unexplained_nan_inf = bool(not np.all(np.isfinite(values)))

    return {
        "paired_lower": float(paired.lower),
        "paired_upper": float(paired.upper),
        "paired_width": paired_width,
        "separate_lower": float(separate.lower),
        "separate_upper": float(separate.upper),
        "separate_width": separate_width,
        "width_ratio": float(width_ratio),
        "percent_width_reduction": float(percentage_reduction),
        "degenerate_separate_width": bool(degenerate_separate),
        "positive_certification": positive,
        "negative_certification": negative,
        "unresolved": unresolved,
        "finite_truth_contained": finite_truth_contained,
        "paired_within_separate": paired_within_separate,
        "lower_le_upper": lower_le_upper,
        "unexplained_nan_inf": unexplained_nan_inf,
    }


def enforce_interval_gates(
    metrics: dict[str, Any],
    *,
    count_contains_m: bool,
    identifiers: dict[str, Any],
) -> None:
    checks = {
        "realized count absent from count set": count_contains_m,
        "finite-cohort containment failure": metrics["finite_truth_contained"],
        "paired interval not within separate interval": metrics["paired_within_separate"],
        "lower endpoint greater than upper endpoint": metrics["lower_le_upper"],
        "unexplained NaN/Inf in interval output": not metrics["unexplained_nan_inf"],
    }
    for message, passed in checks.items():
        if not passed:
            raise SimulationValidationError(message, dict(identifiers, metrics=metrics))
