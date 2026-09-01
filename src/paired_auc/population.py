"""Population functionals and empirical plug-in endpoints from Sections 4--5."""

from dataclasses import dataclass

import numpy as np

from .core import InputValidationError, PairedAUCError, paired_rank_contrasts, validate_observed_labels, validate_score_pair


class UndefinedPopulationFunctionalError(PairedAUCError):
    """Population/plugin expression is undefined under supplied components."""


@dataclass(frozen=True)
class PopulationBounds:
    lower: float
    upper: float
    prevalence: float
    tau: float
    lower_tail_integral: float
    upper_tail_integral: float

    @property
    def width(self):
        return self.upper - self.lower


@dataclass(frozen=True)
class PluginBounds:
    lower: float
    upper: float
    prevalence: float
    rho_hat: float
    p1_hat: float
    mu1_hat: float
    tau_raw: float
    tau_hat: float
    projection_used: bool

    @property
    def width(self):
        return self.upper - self.lower


def _validate_mass(mass):
    try:
        value = float(mass)
    except (TypeError, ValueError):
        raise InputValidationError("tail mass must be numeric")
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise InputValidationError("tail mass must lie in [0,1]")
    return value


def trimmed_expectation_bounds(values, mass, probabilities=None):
    """Lower/upper quantile integrals for a finite discrete distribution.

    Fractional probability at the boundary is retained, matching Lemma A.2.
    """

    z = np.asarray(values, dtype=float)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise InputValidationError("values must be a nonempty finite one-dimensional array")
    tau = _validate_mass(mass)
    if probabilities is None:
        probabilities = np.full(z.size, 1.0 / z.size, dtype=float)
    else:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 1 or probabilities.size != z.size:
            raise InputValidationError("probabilities must align with values")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise InputValidationError("probabilities must be finite and nonnegative")
        total = float(np.sum(probabilities))
        if not np.isclose(total, 1.0, atol=1e-12, rtol=1e-12):
            raise InputValidationError("probabilities must sum to one")
        probabilities = probabilities / total

    def allocate(order):
        remaining = tau
        value = 0.0
        for index in order:
            take = min(float(probabilities[index]), remaining)
            value += take * float(z[index])
            remaining -= take
            if remaining <= 1e-15:
                break
        return value

    order = np.argsort(z, kind="mergesort")
    return allocate(order), allocate(order[::-1])


def population_bounds_from_components(rho, p1, mu1, unverified_d_values, prevalence, probabilities=None):
    """Theorem 4.2 evaluated for a finite representation of D | R=0."""

    rho = float(rho)
    p1 = float(p1)
    mu1 = float(mu1)
    prevalence = float(prevalence)
    if not all(np.isfinite(value) for value in (rho, p1, mu1, prevalence)):
        raise InputValidationError("population components must be finite")
    if rho <= 0.0:
        raise UndefinedPopulationFunctionalError("rho must be positive")
    if prevalence <= 0.0 or prevalence >= 1.0:
        raise UndefinedPopulationFunctionalError("prevalence must lie strictly between zero and one")
    if prevalence < p1 - 1e-12 or prevalence > p1 + rho + 1e-12:
        raise InputValidationError("prevalence is incompatible with p1 and rho")
    tau = (prevalence - p1) / rho
    if tau < -1e-12 or tau > 1.0 + 1e-12:
        raise InputValidationError("implied unverified positive fraction is infeasible")
    tau = min(1.0, max(0.0, tau))
    lower_tail, upper_tail = trimmed_expectation_bounds(
        unverified_d_values, tau, probabilities=probabilities
    )
    denominator = prevalence * (1.0 - prevalence)
    return PopulationBounds(
        (mu1 + rho * lower_tail) / denominator,
        (mu1 + rho * upper_tail) / denominator,
        prevalence,
        tau,
        lower_tail,
        upper_tail,
    )


def population_bounds_prevalence_set(rho, p1, mu1, unverified_d_values, prevalences, probabilities=None):
    try:
        pi_values = sorted(set(float(value) for value in prevalences))
    except (TypeError, ValueError):
        raise InputValidationError("prevalences must be an iterable of numeric values")
    if not pi_values:
        raise InputValidationError("prevalence set must not be empty")
    results = tuple(
        population_bounds_from_components(
            rho, p1, mu1, unverified_d_values, pi, probabilities=probabilities
        )
        for pi in pi_values
    )
    return min(item.lower for item in results), max(item.upper for item in results), results


def empirical_mid_distribution_at_observations(scores):
    """Equation (35) at observed scores: (midrank - 1/2) / n."""

    from .core import ascending_midranks, validate_scores

    s = validate_scores(scores)
    return (ascending_midranks(s) - 0.5) / s.size


def empirical_plugin_bounds(scores_a, scores_b, labels, verified, prevalence):
    """Empirical plug-in endpoints in Section 5.

    This is an estimator. It is deliberately separate from the population
    functional above, whose inputs are population components.
    """

    a, b = validate_score_pair(scores_a, scores_b)
    y, r = validate_observed_labels(labels, verified, a.size)
    prevalence = float(prevalence)
    if not np.isfinite(prevalence) or prevalence <= 0.0 or prevalence >= 1.0:
        raise UndefinedPopulationFunctionalError("plug-in prevalence must lie in (0,1)")
    u = int(np.sum(~r))
    if u == 0:
        raise UndefinedPopulationFunctionalError(
            "empirical tail plug-in is undefined when there are no unverified subjects"
        )
    d_hat = paired_rank_contrasts(a, b) / float(a.size)
    rho_hat = u / float(a.size)
    p1_hat = float(np.sum(y[r])) / a.size
    mu1_hat = float(np.dot(y[r], d_hat[r])) / a.size
    tau_raw = (prevalence - p1_hat) / rho_hat
    tau_hat = min(1.0, max(0.0, tau_raw))
    lower_tail, upper_tail = trimmed_expectation_bounds(d_hat[~r], tau_hat)
    denominator = prevalence * (1.0 - prevalence)
    return PluginBounds(
        (mu1_hat + rho_hat * lower_tail) / denominator,
        (mu1_hat + rho_hat * upper_tail) / denominator,
        prevalence,
        rho_hat,
        p1_hat,
        mu1_hat,
        tau_raw,
        tau_hat,
        not np.isclose(tau_raw, tau_hat, atol=0.0, rtol=0.0),
    )
