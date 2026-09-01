#!/usr/bin/env python
"""Calibrate verification-model intercepts and check reproducibility."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import yaml  # noqa: E402
from scipy.special import logit  # noqa: E402

from paired_auc.calibration import (  # noqa: E402
    calibrate_all,
    calibration_values_equal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=CODE_ROOT / "config" / "simulation_protocol.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_ROOT / "results" / "calibration",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary = calibrate_all(args.protocol)
    reproducibility_start = time.perf_counter()
    repeated = calibrate_all(args.protocol)
    reproducibility_seconds = time.perf_counter() - reproducibility_start
    reproducible = calibration_values_equal(primary.table, repeated.table)

    table = primary.table
    numerical = table[table["calibration_method"] == "numerical"]
    analytic = table[table["calibration_method"] == "analytic"]
    numerical_expected = 36
    numerical_passed = int((numerical["status"] == "PASS").sum())
    numerical_failed = int((numerical["status"] != "PASS").sum())
    analytic_passed = bool(
        len(analytic) == 18
        and (analytic["status"] == "PASS").all()
        and np.array_equal(
            analytic["gamma_0"].to_numpy(),
            logit(analytic["target_verification_rate"].to_numpy()),
        )
    )

    table_path = args.output_dir / "gamma0_calibration.csv"
    metadata_path = args.output_dir / "calibration_run_metadata.json"
    table.to_csv(table_path, index=False, lineterminator="\n")
    metadata = {
        "purpose": "verification-intercept calibration",
        "protocol": str(args.protocol.resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
        "rows": int(len(table)),
        "numerical_calibrations_expected": numerical_expected,
        "numerical_calibrations_passed": numerical_passed,
        "numerical_calibrations_failed": numerical_failed,
        "mcar_analytic_rows": int(len(analytic)),
        "mcar_analytic_passed": analytic_passed,
        "maximum_validation_rate_error": float(
            numerical["absolute_validation_error"].max()
        ),
        "primary_calibration_wall_seconds": primary.wall_seconds,
        "reproducibility_check_wall_seconds": reproducibility_seconds,
        "fixed_seed_reproducible": reproducible,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Wrote {table_path}")
    print(f"Wrote {metadata_path}")

    passed = (
        len(numerical) == numerical_expected
        and numerical_passed == numerical_expected
        and numerical_failed == 0
        and analytic_passed
        and reproducible
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
