#!/usr/bin/env python
"""Run or resume the 162-cell simulation from local checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import scipy  # noqa: E402

from paired_auc.metrics import SimulationValidationError  # noqa: E402
from paired_auc.simulation import (  # noqa: E402
    enumerate_cells,
    gamma_for_cell,
    load_gamma0_table,
    load_protocol,
    run_cell_worker,
    save_failure,
)


REGIMES = ("exact", "bounded", "unrestricted")
DEFAULT_WORKERS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=CODE_ROOT / "config" / "simulation_protocol.yaml",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=CODE_ROOT / "results" / "calibration" / "gamma0_calibration.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_ROOT / "results" / "simulation",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def _event(log_path: Path, event: str, **values: Any) -> None:
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def normalize_frame_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Give all cell checkpoints one stable nullable Parquet schema."""

    frame = frame.copy()
    boolean_columns = {
        "target_undefined",
        "cc_undefined",
    }
    integer_columns = {
        "cell_id",
        "n",
        "replicate_id",
        "seed_base",
        "seed_cell_id",
        "seed_replicate_id",
        "total_positives_m",
        "total_negatives",
        "verified_count",
        "verified_positives",
        "verified_negatives",
        "unverified_count",
    }
    string_columns = {"mechanism", "seed_scheme", "replicate_status", "warning_code"}
    for regime in REGIMES:
        integer_columns.update(
            {
                f"{regime}_count_set_min",
                f"{regime}_count_set_max",
                f"{regime}_count_set_size",
            }
        )
        boolean_columns.update(
            {
                f"{regime}_count_contains_m",
                f"{regime}_degenerate_separate_width",
                f"{regime}_positive_certification",
                f"{regime}_negative_certification",
                f"{regime}_unresolved",
                f"{regime}_finite_truth_contained",
                f"{regime}_paired_within_separate",
                f"{regime}_lower_le_upper",
                f"{regime}_unexplained_nan_inf",
            }
        )
    for column in frame.columns:
        if column in boolean_columns:
            frame[column] = frame[column].astype("boolean")
        elif column in integer_columns:
            frame[column] = frame[column].astype("Int64")
        elif column in string_columns:
            frame[column] = frame[column].astype("string")
        else:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    return frame


def cell_gate_counts(
    frame: pd.DataFrame,
    *,
    expected_cell_id: int,
    repetitions: int,
    simulation_base_seed: int,
) -> dict[str, int]:
    expected_ids = set(range(repetitions))
    actual_ids = set(int(value) for value in frame["replicate_id"].dropna())
    target_defined = frame[~frame["target_undefined"].astype(bool)]
    counts = {
        "wrong_row_count": int(len(frame) != repetitions),
        "unexpected_cell_ids": int(
            set(int(value) for value in frame["cell_id"].dropna()) != {expected_cell_id}
        ),
        "wrong_seed_base": int(
            set(int(value) for value in frame["seed_base"].dropna())
            != {simulation_base_seed}
        ),
        "duplicate_replicate_ids": int(frame["replicate_id"].duplicated().sum()),
        "missing_replicate_ids": len(expected_ids - actual_ids),
        "unexpected_replicate_ids": len(actual_ids - expected_ids),
        "failed_replicates": int(
            (~frame["replicate_status"].isin(["PASS", "target_undefined"])).sum()
        ),
        "containment_violations": 0,
        "paired_within_separate_violations": 0,
        "lower_greater_upper_violations": 0,
        "unexplained_nan_inf": 0,
        "realized_count_membership_violations": 0,
    }
    for regime in REGIMES:
        counts["containment_violations"] += int(
            (~target_defined[f"{regime}_finite_truth_contained"].astype(bool)).sum()
        )
        counts["paired_within_separate_violations"] += int(
            (~target_defined[f"{regime}_paired_within_separate"].astype(bool)).sum()
        )
        counts["lower_greater_upper_violations"] += int(
            (~target_defined[f"{regime}_lower_le_upper"].astype(bool)).sum()
        )
        counts["unexplained_nan_inf"] += int(
            target_defined[f"{regime}_unexplained_nan_inf"].astype(bool).sum()
        )
        counts["realized_count_membership_violations"] += int(
            (~target_defined[f"{regime}_count_contains_m"].astype(bool)).sum()
        )
        ratio_missing = target_defined[f"{regime}_width_ratio"].isna()
        degenerate = target_defined[f"{regime}_degenerate_separate_width"].astype(bool)
        counts["unexplained_nan_inf"] += int((ratio_missing != degenerate).sum())
    return counts


def validate_cell_frame(
    frame: pd.DataFrame,
    *,
    expected_cell_id: int,
    repetitions: int,
    simulation_base_seed: int,
) -> dict[str, int]:
    counts = cell_gate_counts(
        frame,
        expected_cell_id=expected_cell_id,
        repetitions=repetitions,
        simulation_base_seed=simulation_base_seed,
    )
    if any(counts.values()):
        raise SimulationValidationError(
            f"simulation cell {expected_cell_id} failed validation", counts
        )
    return counts


def _write_cell_checkpoint(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(
        temporary,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    temporary.replace(path)


def _merge_cell_files(cell_paths: list[Path], output_path: Path) -> None:
    temporary = output_path.with_suffix(".tmp.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    expected_schema = None
    try:
        for path in cell_paths:
            table = pq.read_table(path)
            if writer is None:
                expected_schema = table.schema
                writer = pq.ParquetWriter(
                    temporary,
                    expected_schema,
                    compression="zstd",
                )
            elif table.schema != expected_schema:
                table = table.cast(expected_schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("no cell files available for simulation merge")
    temporary.replace(output_path)


def _sanity_summary(results_path: Path) -> pd.DataFrame:
    columns = [
        "mechanism",
        "target_verification_rate",
        "achieved_verification_fraction",
        "target_undefined",
        "cc_undefined",
    ]
    for regime in REGIMES:
        columns.extend(
            [
                f"{regime}_paired_width",
                f"{regime}_separate_width",
                f"{regime}_width_ratio",
                f"{regime}_percent_width_reduction",
                f"{regime}_positive_certification",
                f"{regime}_unresolved",
            ]
        )
    data = pd.read_parquet(results_path, columns=columns)
    rows: list[dict[str, Any]] = []
    groups = [("overall", None, None, data)] + [
        ("mechanism_target", str(mechanism), float(target), part)
        for (mechanism, target), part in data.groupby(
            ["mechanism", "target_verification_rate"], sort=True
        )
    ]
    for grouping, mechanism, target, part in groups:
        defined = part[~part["target_undefined"].astype(bool)]
        for regime in REGIMES:
            ratio = defined[f"{regime}_width_ratio"]
            reduction = defined[f"{regime}_percent_width_reduction"]
            rows.append(
                {
                    "grouping": grouping,
                    "mechanism": mechanism,
                    "target_verification_rate": target,
                    "count_regime": regime,
                    "datasets": int(len(part)),
                    "target_defined": int(len(defined)),
                    "target_undefined_fraction": float(
                        part["target_undefined"].astype(bool).mean()
                    ),
                    "cc_undefined_fraction": float(
                        part["cc_undefined"].astype(bool).mean()
                    ),
                    "mean_achieved_verification_rate": float(
                        part["achieved_verification_fraction"].mean()
                    ),
                    "mean_paired_width": float(
                        defined[f"{regime}_paired_width"].mean()
                    ),
                    "mean_separate_width": float(
                        defined[f"{regime}_separate_width"].mean()
                    ),
                    "mean_width_ratio": float(ratio.mean()),
                    "width_ratio_valid": int(ratio.notna().sum()),
                    "mean_percent_width_reduction": float(reduction.mean()),
                    "percent_reduction_valid": int(reduction.notna().sum()),
                    "positive_certification_rate": float(
                        defined[f"{regime}_positive_certification"].astype(bool).mean()
                    ),
                    "unresolved_rate": float(
                        defined[f"{regime}_unresolved"].astype(bool).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    protocol = load_protocol(args.protocol)
    settings = protocol["simulation"]
    repetitions = int(settings["repetitions_per_dgm_cell"])
    expected_cells = int(settings["dgm_cells"])
    expected_datasets = int(settings["simulated_dataset_count"])
    expected_analyses = int(settings["paired_count_regime_analysis_count"])
    base_seed = int(protocol["random_streams"]["base_seed"])
    if (repetitions, expected_cells, expected_datasets, expected_analyses) != (
        2000,
        162,
        324000,
        972000,
    ):
        raise SimulationValidationError("simulation dimensions do not match the specification")
    if base_seed != 20260901:
        raise SimulationValidationError("simulation base seed does not match the specification")

    cells = enumerate_cells(protocol)
    gamma_lookup = load_gamma0_table(args.calibration)
    raw_dir = args.output_dir / "raw" / "cells"
    summary_dir = args.output_dir / "summary"
    logs_dir = args.output_dir / "logs"
    event_log = logs_dir / "simulation_events.jsonl"
    failure_path = logs_dir / "minimal_failure_case.json"
    full_results_path = args.output_dir / "simulation_results.parquet"
    sanity_path = args.output_dir / "simulation_summary_sanity.csv"
    metadata_path = args.output_dir / "simulation_run_metadata.json"
    manifest_path = args.output_dir / "simulation_cell_manifest.csv"
    start_wall = time.perf_counter()
    start_utc = datetime.now(timezone.utc)
    _event(
        event_log,
        "simulation_start_or_resume",
        workers=args.workers,
        simulation_base_seed=base_seed,
        repetitions_per_cell=repetitions,
    )

    completed_paths: dict[int, Path] = {}
    cell_worker_walls: dict[int, float] = {}
    try:
        for cell in cells:
            path = raw_dir / f"cell_{cell.cell_id:03d}.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                validate_cell_frame(
                    existing,
                    expected_cell_id=cell.cell_id,
                    repetitions=repetitions,
                    simulation_base_seed=base_seed,
                )
                completed_paths[cell.cell_id] = path
                _event(event_log, "cell_checkpoint_reused", cell_id=cell.cell_id)

        pending = [cell for cell in cells if cell.cell_id not in completed_paths]
        print(
            f"Simulation start/resume: {len(completed_paths)} cells complete, "
            f"{len(pending)} pending, workers={args.workers}",
            flush=True,
        )
        payload_by_cell = {
            cell.cell_id: (
                protocol,
                asdict(cell),
                list(range(repetitions)),
                gamma_for_cell(gamma_lookup, cell),
                base_seed,
            )
            for cell in pending
        }
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_cell_worker, payload): cell_id
                for cell_id, payload in payload_by_cell.items()
            }
            for future in as_completed(futures):
                cell_id = futures[future]
                try:
                    rows, worker_wall = future.result()
                    frame = normalize_frame_types(pd.DataFrame(rows))
                    validate_cell_frame(
                        frame,
                        expected_cell_id=cell_id,
                        repetitions=repetitions,
                        simulation_base_seed=base_seed,
                    )
                    path = raw_dir / f"cell_{cell_id:03d}.parquet"
                    _write_cell_checkpoint(frame, path)
                    completed_paths[cell_id] = path
                    cell_worker_walls[cell_id] = float(worker_wall)
                    _event(
                        event_log,
                        "cell_completed",
                        cell_id=cell_id,
                        rows=len(frame),
                        worker_wall_seconds=worker_wall,
                    )
                    print(
                        f"Completed cell {cell_id:03d}; "
                        f"progress {len(completed_paths)}/{expected_cells}",
                        flush=True,
                    )
                except BaseException as error:
                    for other in futures:
                        other.cancel()
                    save_failure(failure_path, error)
                    _event(
                        event_log,
                        "simulation_failure",
                        cell_id=cell_id,
                        error_type=type(error).__name__,
                        message=str(error),
                    )
                    raise

        if set(completed_paths) != set(range(expected_cells)):
            missing = sorted(set(range(expected_cells)) - set(completed_paths))
            raise SimulationValidationError(
                "simulation ended with missing cell checkpoints", {"missing_cells": missing}
            )

        manifest_rows: list[dict[str, Any]] = []
        aggregate = {
            "completed_datasets": 0,
            "target_undefined": 0,
            "cc_undefined": 0,
            "failed_replicates": 0,
            "containment_violations": 0,
            "paired_within_separate_violations": 0,
            "lower_greater_upper_violations": 0,
            "unexplained_nan_inf": 0,
            "realized_count_membership_violations": 0,
            "duplicate_replicate_ids": 0,
            "missing_replicate_ids": 0,
            "unexpected_replicate_ids": 0,
            "unexpected_cell_ids": 0,
            "wrong_seed_base": 0,
            "wrong_row_count": 0,
        }
        ordered_paths = [completed_paths[cell_id] for cell_id in range(expected_cells)]
        for cell_id, path in enumerate(ordered_paths):
            frame = pd.read_parquet(path)
            counts = validate_cell_frame(
                frame,
                expected_cell_id=cell_id,
                repetitions=repetitions,
                simulation_base_seed=base_seed,
            )
            aggregate["completed_datasets"] += len(frame)
            aggregate["target_undefined"] += int(
                frame["target_undefined"].astype(bool).sum()
            )
            aggregate["cc_undefined"] += int(frame["cc_undefined"].astype(bool).sum())
            for key in counts:
                aggregate[key] += counts[key]
            manifest_rows.append(
                {
                    "cell_id": cell_id,
                    "path": str(path.relative_to(args.output_dir)),
                    "rows": len(frame),
                    "replicate_id_min": int(frame["replicate_id"].min()),
                    "replicate_id_max": int(frame["replicate_id"].max()),
                    "target_undefined": int(
                        frame["target_undefined"].astype(bool).sum()
                    ),
                    "cc_undefined": int(frame["cc_undefined"].astype(bool).sum()),
                    "worker_wall_seconds_this_run": cell_worker_walls.get(
                        cell_id, np.nan
                    ),
                    "status": "PASS",
                }
            )
        if aggregate["completed_datasets"] != expected_datasets:
            raise SimulationValidationError(
                "simulation dataset count mismatch", aggregate
            )
        hard_gate_keys = [
            key
            for key in aggregate
            if key not in {"completed_datasets", "target_undefined", "cc_undefined"}
        ]
        if any(aggregate[key] for key in hard_gate_keys):
            raise SimulationValidationError("aggregate simulation check failure", aggregate)

        pd.DataFrame(manifest_rows).to_csv(
            manifest_path, index=False, lineterminator="\n"
        )
        _merge_cell_files(ordered_paths, full_results_path)
        sanity = _sanity_summary(full_results_path)
        sanity.to_csv(sanity_path, index=False, lineterminator="\n")

        full_table = pq.ParquetFile(full_results_path)
        if full_table.metadata.num_rows != expected_datasets:
            raise SimulationValidationError("merged Parquet row count mismatch")

        end_utc = datetime.now(timezone.utc)
        wall_seconds = time.perf_counter() - start_wall
        actual_analyses = (expected_datasets - aggregate["target_undefined"]) * 3
        metadata = {
            "protocol": str(args.protocol),
            "calibration": str(args.calibration),
            "simulation_start_utc": start_utc.isoformat(),
            "simulation_end_utc": end_utc.isoformat(),
            "wall_seconds": wall_seconds,
            "workers": args.workers,
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "simulation_base_seed": base_seed,
            "expected_cells": expected_cells,
            "completed_cells": len(completed_paths),
            "repetitions_per_cell": repetitions,
            "expected_datasets": expected_datasets,
            "completed_datasets": aggregate["completed_datasets"],
            "failed_datasets": aggregate["failed_replicates"],
            "target_undefined": aggregate["target_undefined"],
            "cc_undefined": aggregate["cc_undefined"],
            "expected_nominal_count_regime_analyses": expected_analyses,
            "completed_count_regime_analyses": actual_analyses,
            "mean_wall_seconds_per_dataset": wall_seconds / expected_datasets,
            "datasets_per_second": expected_datasets / wall_seconds,
            "mean_cell_worker_wall_seconds": float(
                np.mean(list(cell_worker_walls.values()))
            )
            if cell_worker_walls
            else np.nan,
            "simulation_results_path": str(full_results_path.relative_to(args.output_dir)),
            "simulation_results_rows": full_table.metadata.num_rows,
            "simulation_results_row_groups": full_table.metadata.num_row_groups,
            "simulation_results_size_bytes": full_results_path.stat().st_size,
            **{
                key: value
                for key, value in aggregate.items()
                if key not in {"completed_datasets", "target_undefined", "cc_undefined"}
            },
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _event(
            event_log,
            "simulation_complete",
            completed_datasets=expected_datasets,
            wall_seconds=wall_seconds,
            workers=args.workers,
        )
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
        for path in (
            full_results_path,
            sanity_path,
            metadata_path,
            manifest_path,
            event_log,
        ):
            print(f"Wrote {path}", flush=True)
        return 0
    except BaseException as error:
        if not failure_path.exists():
            save_failure(failure_path, error)
        _event(
            event_log,
            "simulation_aborted",
            error_type=type(error).__name__,
            message=str(error),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
