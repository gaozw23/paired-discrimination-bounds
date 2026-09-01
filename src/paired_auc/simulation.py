"""Simulation engine for the paired-discrimination study."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import expit

from .bounds import (
    compatible_total_positive_counts,
    paired_bounds_count_set,
    paired_bounds_exact_count,
    separate_difference_bounds_count_set,
    separate_difference_bounds_exact_count,
)
from .calibration import (
    generate_full_data_sample,
    load_protocol,
    mechanism_coefficients,
    signal_parameters,
    verification_covariates,
)
from .core import empirical_auc
from .metrics import SimulationValidationError, enforce_interval_gates, interval_metrics


REGIMES = ("exact", "bounded", "unrestricted")


@dataclass(frozen=True)
class CellSpec:
    cell_id: int
    n: int
    pi: float
    eta: float
    target_verification_rate: float
    mechanism: str


def enumerate_cells(protocol: dict[str, Any]) -> list[CellSpec]:
    grid = protocol["parameter_grid"]["order"]
    cells: list[CellSpec] = []
    cell_id = 0
    for n in grid["n"]:
        for pi in grid["pi"]:
            for eta in grid["eta"]:
                for target in grid["target_verification_rate"]:
                    for mechanism in grid["verification_mechanism"]:
                        cells.append(
                            CellSpec(
                                cell_id=cell_id,
                                n=int(n),
                                pi=float(pi),
                                eta=float(eta),
                                target_verification_rate=float(target),
                                mechanism=str(mechanism),
                            )
                        )
                        cell_id += 1
    if len(cells) != 162 or cells[-1].cell_id != 161:
        raise SimulationValidationError("cell enumeration does not match the 162-cell grid")
    return cells


def load_gamma0_table(path: str | Path) -> dict[tuple[float, float, str, float], float]:
    table = pd.read_csv(path)
    required = {
        "pi",
        "eta",
        "mechanism",
        "target_verification_rate",
        "gamma_0",
        "status",
    }
    if not required.issubset(table.columns):
        raise ValueError("calibration table lacks required columns")
    if not bool((table["status"] == "PASS").all()):
        raise SimulationValidationError("calibration table contains failed rows")
    result: dict[tuple[float, float, str, float], float] = {}
    for row in table.itertuples(index=False):
        key = (
            float(row.pi),
            float(row.eta),
            str(row.mechanism),
            float(row.target_verification_rate),
        )
        if key in result:
            raise SimulationValidationError(f"duplicate calibration key {key}")
        result[key] = float(row.gamma_0)
    if len(result) != 54:
        raise SimulationValidationError("calibration lookup must contain 54 cell-level rows")
    return result


def _blank_regime_fields(prefix: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "count_set_min": np.nan,
        "count_set_max": np.nan,
        "count_set_size": 0,
        "count_contains_m": False,
        "paired_lower": np.nan,
        "paired_upper": np.nan,
        "paired_width": np.nan,
        "separate_lower": np.nan,
        "separate_upper": np.nan,
        "separate_width": np.nan,
        "width_ratio": np.nan,
        "percent_width_reduction": np.nan,
        "degenerate_separate_width": False,
        "positive_certification": False,
        "negative_certification": False,
        "unresolved": False,
        "finite_truth_contained": False,
        "paired_within_separate": False,
        "lower_le_upper": False,
        "unexplained_nan_inf": False,
    }
    return {f"{prefix}_{key}": value for key, value in fields.items()}


def _analyze_regime(
    *,
    regime: str,
    counts: tuple[int, ...],
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    observed_labels: np.ndarray,
    verified: np.ndarray,
    realized_m: int,
    delta_full_n: float,
    tolerance: float,
    identifiers: dict[str, Any],
) -> dict[str, Any]:
    if regime == "exact":
        paired = paired_bounds_exact_count(
            scores_a, scores_b, observed_labels, verified, realized_m
        )
        separate = separate_difference_bounds_exact_count(
            scores_a, scores_b, observed_labels, verified, realized_m
        )
    else:
        paired = paired_bounds_count_set(
            scores_a, scores_b, observed_labels, verified, counts
        )
        separate = separate_difference_bounds_count_set(
            scores_a, scores_b, observed_labels, verified, counts
        )
    count_contains_m = realized_m in counts
    calculated = interval_metrics(
        paired, separate, delta_full_n, tolerance=tolerance
    )
    enforce_interval_gates(
        calculated,
        count_contains_m=count_contains_m,
        identifiers=dict(identifiers, regime=regime, counts=counts),
    )
    values: dict[str, Any] = {
        "count_set_min": min(counts),
        "count_set_max": max(counts),
        "count_set_size": len(counts),
        "count_contains_m": count_contains_m,
        **calculated,
    }
    return {f"{regime}_{key}": value for key, value in values.items()}


def run_replicate(
    *,
    protocol: dict[str, Any],
    cell: CellSpec,
    replicate_id: int,
    gamma_0: float,
    base_seed: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    if replicate_id < 0:
        raise ValueError("replicate_id must be nonnegative")
    delta_a, delta_b = signal_parameters(protocol)
    replicate_seed = np.random.SeedSequence(
        [int(base_seed), int(cell.cell_id), int(replicate_id)]
    )
    data_seed, verification_seed = replicate_seed.spawn(2)
    y, scores_a, scores_b = generate_full_data_sample(
        size=cell.n,
        pi=cell.pi,
        eta=cell.eta,
        delta_a=delta_a,
        delta_b=delta_b,
        seed_sequence=data_seed,
    )
    mean_score, absolute_difference = verification_covariates(scores_a, scores_b)
    gamma_y, gamma_m, gamma_g = mechanism_coefficients(protocol, cell.mechanism)
    probabilities = expit(
        gamma_0
        + gamma_y * y
        + gamma_m * mean_score
        + gamma_g * absolute_difference
    )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise SimulationValidationError(
            "illegal verification probabilities",
            {"cell_id": cell.cell_id, "replicate_id": replicate_id},
        )
    verification_rng = np.random.default_rng(verification_seed)
    verified = verification_rng.random(cell.n) < probabilities
    observed_labels = np.where(verified, y.astype(float), np.nan)

    m = int(np.sum(y))
    verified_count = int(np.sum(verified))
    verified_positives = int(np.sum(y[verified]))
    verified_negatives = int(verified_count - verified_positives)
    target_undefined = m == 0 or m == cell.n
    identifiers = {"cell_id": cell.cell_id, "replicate_id": replicate_id}
    row: dict[str, Any] = {
        **asdict(cell),
        "replicate_id": int(replicate_id),
        "seed_base": int(base_seed),
        "seed_cell_id": int(cell.cell_id),
        "seed_replicate_id": int(replicate_id),
        "seed_scheme": "SeedSequence([base_seed,cell_id,replicate_id]).spawn(2)",
        "gamma_0": float(gamma_0),
        "gamma_y": gamma_y,
        "gamma_m": gamma_m,
        "gamma_g": gamma_g,
        "total_positives_m": m,
        "total_negatives": int(cell.n - m),
        "verified_count": verified_count,
        "achieved_verification_fraction": verified_count / float(cell.n),
        "verified_positives": verified_positives,
        "verified_negatives": verified_negatives,
        "unverified_count": int(cell.n - verified_count),
        "delta_population": float(protocol["dgm"]["auc_targets"]["delta_population"]),
        "auc_a_full": np.nan,
        "auc_b_full": np.nan,
        "delta_full_n": np.nan,
        "target_undefined": bool(target_undefined),
        "auc_a_cc": np.nan,
        "auc_b_cc": np.nan,
        "delta_cc": np.nan,
        "cc_undefined": True,
        "cc_error_vs_full": np.nan,
        "replicate_status": "target_undefined" if target_undefined else "PASS",
        "warning_code": "TARGET_UNDEFINED" if target_undefined else "OK",
    }
    for regime in REGIMES:
        row.update(_blank_regime_fields(regime))

    if not target_undefined:
        auc_a_full = empirical_auc(scores_a, y)
        auc_b_full = empirical_auc(scores_b, y)
        delta_full_n = auc_a_full - auc_b_full
        row.update(
            {
                "auc_a_full": auc_a_full,
                "auc_b_full": auc_b_full,
                "delta_full_n": delta_full_n,
            }
        )
        if verified_positives > 0 and verified_negatives > 0:
            auc_a_cc = empirical_auc(scores_a[verified], y[verified])
            auc_b_cc = empirical_auc(scores_b[verified], y[verified])
            delta_cc = auc_a_cc - auc_b_cc
            row.update(
                {
                    "auc_a_cc": auc_a_cc,
                    "auc_b_cc": auc_b_cc,
                    "delta_cc": delta_cc,
                    "cc_undefined": False,
                    "cc_error_vs_full": delta_cc - delta_full_n,
                }
            )
        else:
            row["warning_code"] = "CC_UNDEFINED"

        compatible = compatible_total_positive_counts(
            observed_labels, verified, cell.n
        )
        radius = int(math.ceil(0.03 * cell.n))
        bounded = tuple(
            count
            for count in compatible
            if m - radius <= count <= m + radius
        )
        regime_counts = {
            "exact": (m,),
            "bounded": bounded,
            "unrestricted": compatible,
        }
        tolerance = float(protocol["numerics"]["implementation_tolerance"])
        for regime, counts in regime_counts.items():
            row.update(
                _analyze_regime(
                    regime=regime,
                    counts=counts,
                    scores_a=scores_a,
                    scores_b=scores_b,
                    observed_labels=observed_labels,
                    verified=verified,
                    realized_m=m,
                    delta_full_n=delta_full_n,
                    tolerance=tolerance,
                    identifiers=identifiers,
                )
            )

    row["replicate_runtime_seconds"] = time.perf_counter() - start
    return row


def run_cell(
    *,
    protocol: dict[str, Any],
    cell: CellSpec,
    replicate_ids: Iterable[int],
    gamma_0: float,
    base_seed: int,
) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    rows = [
        run_replicate(
            protocol=protocol,
            cell=cell,
            replicate_id=int(replicate_id),
            gamma_0=gamma_0,
            base_seed=base_seed,
        )
        for replicate_id in replicate_ids
    ]
    return rows, time.perf_counter() - start


def run_cell_worker(payload: tuple[dict[str, Any], dict[str, Any], list[int], float, int]):
    protocol, cell_values, replicate_ids, gamma_0, base_seed = payload
    return run_cell(
        protocol=protocol,
        cell=CellSpec(**cell_values),
        replicate_ids=replicate_ids,
        gamma_0=gamma_0,
        base_seed=base_seed,
    )


def gamma_for_cell(
    gamma_lookup: dict[tuple[float, float, str, float], float], cell: CellSpec
) -> float:
    key = (cell.pi, cell.eta, cell.mechanism, cell.target_verification_rate)
    try:
        return gamma_lookup[key]
    except KeyError as error:
        raise SimulationValidationError(f"missing gamma_0 for cell {cell.cell_id}: {key}") from error


def deterministic_frame_signature(frame: pd.DataFrame) -> str:
    """Hash all deterministic fields; runtime is intentionally excluded."""

    deterministic = frame.drop(columns=["replicate_runtime_seconds"], errors="ignore").copy()
    deterministic = deterministic.sort_values(["cell_id", "replicate_id"], kind="stable")
    deterministic.reset_index(drop=True, inplace=True)
    payload = deterministic.to_json(orient="split", double_precision=15, index=False)
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_failure(path: str | Path, error: BaseException) -> None:
    record = {
        "error_type": type(error).__name__,
        "message": str(error),
        "record": getattr(error, "record", {}),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
