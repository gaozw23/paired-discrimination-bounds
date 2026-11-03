"""Exact external-binomial plans for empirical prevalence sensitivity."""
from __future__ import annotations
import math
from numbers import Integral
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta, binom

SIGN_TOLERANCE = 1e-12
STATUSES = ("valid", "undefined", "incompatible", "empty_domain", "numerical_failure")

def clopper_pearson(k: int, n: int, alpha: float = .05) -> tuple[float, float]:
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("external N must be a positive integer")
    if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= k <= n:
        raise ValueError("external K must be an integer in [0,N]")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    lo = 0. if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1. if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi <= 1):
        raise ArithmeticError("invalid beta quantile in CP construction")
    return lo, hi


def binomial_plan(n: int, pi_true: float) -> tuple[list[dict], dict]:
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("external N must be a positive integer")
    if not math.isfinite(pi_true) or not 0 < pi_true < 1:
        raise ValueError("pi_true must be strictly in (0,1)")
    k_values = np.arange(n + 1)
    logw = binom.logpmf(k_values, n, pi_true)
    if not np.isfinite(logw).all():
        raise ArithmeticError("finite log probabilities required for all K")
    w = np.exp(logw)
    underflow = w == 0.
    logmass = float(logsumexp(logw))
    underflow_logmass = float(logsumexp(logw[underflow])) if underflow.any() else None
    rows = []
    for k, lw, weight, uf in zip(k_values, logw, w, underflow):
        lo, hi = clopper_pearson(int(k), n)
        rows.append({"N": n, "K": int(k), "pi_true": float(pi_true),
                     "log_weight": float(lw), "weight": float(weight),
                     "weight_underflow": bool(uf), "cp_lower": lo,
                     "cp_upper": hi, "cp_contains_pi_true": lo <= pi_true <= hi})
    return rows, {"N": n, "pi_true": float(pi_true), "K_count": n + 1,
                  "raw_log_mass": logmass, "raw_mass": math.exp(logmass),
                  "raw_mass_error": math.expm1(logmass),
                  "float_weight_sum": math.fsum(float(x) for x in w),
                  "underflow_count": int(underflow.sum()),
                  "underflow_log_mass": underflow_logmass,
                  "zero_underflow_mass": not bool(underflow.any()),
                  "renormalized": False}
