"""Verification-intercept calibration for the simulation design."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import norm


@dataclass(frozen=True)
class CalibrationRun:
    table: pd.DataFrame
    wall_seconds: float


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if protocol["parameter_grid"]["dgm_cell_count"] != 162:
        raise ValueError("protocol must contain 162 DGM cells")
    return protocol


def signal_parameters(protocol: dict[str, Any]) -> tuple[float, float]:
    targets = protocol["dgm"]["auc_targets"]
    delta_a = math.sqrt(2.0) * float(norm.ppf(targets["auc_a"]))
    delta_b = math.sqrt(2.0) * float(norm.ppf(targets["auc_b"]))
    tolerance = float(protocol["numerics"]["implementation_tolerance"])
    if abs(float(norm.cdf(delta_a / math.sqrt(2.0))) - targets["auc_a"]) > tolerance:
        raise RuntimeError("AUC_A signal identity failed")
    if abs(float(norm.cdf(delta_b / math.sqrt(2.0))) - targets["auc_b"]) > tolerance:
        raise RuntimeError("AUC_B signal identity failed")
    return delta_a, delta_b


def generate_full_data_sample(
    *,
    size: int,
    pi: float,
    eta: float,
    delta_a: float,
    delta_b: float,
    seed_sequence: np.random.SeedSequence,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate labels and paired scores from the specified normal model."""

    rng = np.random.default_rng(seed_sequence)
    y = rng.binomial(1, pi, size=size).astype(np.int8)
    z_a = rng.standard_normal(size)
    z_independent = rng.standard_normal(size)
    s_a = delta_a * y + z_a
    s_b = delta_b * y + eta * z_a + math.sqrt(1.0 - eta * eta) * z_independent
    return y, s_a, s_b


def verification_covariates(s_a: np.ndarray, s_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (s_a + s_b) / 2.0, np.abs(s_a - s_b)


def mechanism_coefficients(protocol: dict[str, Any], mechanism: str) -> tuple[float, float, float]:
    spec = protocol["verification_model"]["mechanisms"][mechanism]
    gamma_y = math.log(3.0) if spec["gamma_y"] == "log(3)" else float(spec["gamma_y"])
    return gamma_y, float(spec["gamma_m"]), float(spec["gamma_g"])


def calibrate_all(protocol_path: str | Path) -> CalibrationRun:
    protocol = load_protocol(protocol_path)
    calibration = protocol["calibration"]
    grid = protocol["parameter_grid"]["order"]
    delta_a, delta_b = signal_parameters(protocol)
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []

    for pi_index, pi in enumerate(grid["pi"]):
        for eta_index, eta in enumerate(grid["eta"]):
            root_seed = np.random.SeedSequence(
                [int(calibration["base_seed"]), pi_index, eta_index]
            )
            calibration_seed, validation_seed = root_seed.spawn(2)
            y_cal, s_a_cal, s_b_cal = generate_full_data_sample(
                size=int(calibration["calibration_sample_size"]),
                pi=float(pi),
                eta=float(eta),
                delta_a=delta_a,
                delta_b=delta_b,
                seed_sequence=calibration_seed,
            )
            y_val, s_a_val, s_b_val = generate_full_data_sample(
                size=int(calibration["independent_validation_sample_size"]),
                pi=float(pi),
                eta=float(eta),
                delta_a=delta_a,
                delta_b=delta_b,
                seed_sequence=validation_seed,
            )
            m_cal, g_cal = verification_covariates(s_a_cal, s_b_cal)
            m_val, g_val = verification_covariates(s_a_val, s_b_val)

            for mechanism in grid["verification_mechanism"]:
                gamma_y, gamma_m, gamma_g = mechanism_coefficients(protocol, mechanism)
                linear_cal = gamma_y * y_cal + gamma_m * m_cal + gamma_g * g_cal
                linear_val = gamma_y * y_val + gamma_m * m_val + gamma_g * g_val
                for target in grid["target_verification_rate"]:
                    target = float(target)
                    combination_start = time.perf_counter()
                    if mechanism == "MCAR":
                        gamma_0 = float(logit(target))
                        calibration_rate = float(expit(gamma_0))
                        validation_rate = calibration_rate
                        method = "analytic"
                        root_residual = calibration_rate - target
                        probability_min = calibration_rate
                        probability_max = calibration_rate
                    else:
                        solver = calibration["root_solver"]

                        def objective(intercept: float) -> float:
                            return float(np.mean(expit(intercept + linear_cal)) - target)

                        lower, upper = (float(value) for value in solver["bracket"])
                        if objective(lower) * objective(upper) >= 0.0:
                            raise RuntimeError(
                                f"calibration root is not bracketed for pi={pi}, eta={eta}, "
                                f"mechanism={mechanism}, target={target}"
                            )
                        gamma_0 = float(
                            brentq(
                                objective,
                                lower,
                                upper,
                                xtol=float(solver["xtol"]),
                                rtol=float(solver["rtol"]),
                                maxiter=int(solver["maximum_iterations"]),
                            )
                        )
                        probabilities_cal = expit(gamma_0 + linear_cal)
                        probabilities_val = expit(gamma_0 + linear_val)
                        calibration_rate = float(np.mean(probabilities_cal))
                        validation_rate = float(np.mean(probabilities_val))
                        root_residual = calibration_rate - target
                        probability_min = float(
                            min(np.min(probabilities_cal), np.min(probabilities_val))
                        )
                        probability_max = float(
                            max(np.max(probabilities_cal), np.max(probabilities_val))
                        )
                        method = "numerical"

                    validation_error = abs(validation_rate - target)
                    finite = bool(
                        np.all(
                            np.isfinite(
                                [
                                    gamma_0,
                                    calibration_rate,
                                    validation_rate,
                                    root_residual,
                                    probability_min,
                                    probability_max,
                                ]
                            )
                        )
                    )
                    probability_legal = bool(
                        0.0 <= probability_min <= probability_max <= 1.0
                    )
                    root_ok = bool(
                        abs(root_residual)
                        <= float(calibration["root_solver"]["calibration_mean_absolute_tolerance"])
                    )
                    validation_ok = bool(
                        validation_error
                        <= float(
                            str(calibration["independent_validation"]["pass_rule"])
                            .split("<=")[-1]
                            .strip()
                        )
                    )
                    status = "PASS" if finite and probability_legal and root_ok and validation_ok else "FAIL"
                    rows.append(
                        {
                            "pi": float(pi),
                            "eta": float(eta),
                            "mechanism": mechanism,
                            "target_verification_rate": target,
                            "gamma_0": gamma_0,
                            "gamma_y": gamma_y,
                            "gamma_m": gamma_m,
                            "gamma_g": gamma_g,
                            "calibration_method": method,
                            "calibration_achieved_rate": calibration_rate,
                            "root_residual": root_residual,
                            "validation_achieved_rate": validation_rate,
                            "absolute_validation_error": validation_error,
                            "probability_min": probability_min,
                            "probability_max": probability_max,
                            "finite_values": finite,
                            "probabilities_legal": probability_legal,
                            "status": status,
                            "combination_runtime_seconds": time.perf_counter()
                            - combination_start,
                            "calibration_base_seed": int(calibration["base_seed"]),
                            "calibration_sample_size": int(
                                calibration["calibration_sample_size"]
                            ),
                            "validation_sample_size": int(
                                calibration["independent_validation_sample_size"]
                            ),
                        }
                    )

    table = pd.DataFrame(rows).sort_values(
        ["pi", "eta", "mechanism", "target_verification_rate"],
        kind="stable",
    )
    table.reset_index(drop=True, inplace=True)
    return CalibrationRun(table=table, wall_seconds=time.perf_counter() - start)


def calibration_values_equal(first: pd.DataFrame, second: pd.DataFrame) -> bool:
    deterministic_columns = [
        column
        for column in first.columns
        if column != "combination_runtime_seconds"
    ]
    try:
        pd.testing.assert_frame_equal(
            first[deterministic_columns],
            second[deterministic_columns],
            check_exact=True,
            check_dtype=True,
        )
    except AssertionError:
        return False
    return True
