import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from paired_auc.calibration import load_protocol, signal_parameters
from paired_auc.simulation import CellSpec, enumerate_cells, run_replicate
from scripts.run_simulation import normalize_frame_types, validate_cell_frame


def protocol():
    return load_protocol("config/simulation_protocol.yaml")


def deterministic_projection(row):
    return {
        key: value
        for key, value in row.items()
        if key != "replicate_runtime_seconds"
    }


def test_grid_has_162_unique_cells():
    cells = enumerate_cells(protocol())
    assert len(cells) == 162
    assert [cell.cell_id for cell in cells] == list(range(162))
    assert len({(cell.n, cell.pi, cell.eta, cell.target_verification_rate, cell.mechanism) for cell in cells}) == 162


def test_signal_parameters_recover_target_aucs():
    p = protocol()
    delta_a, delta_b = signal_parameters(p)
    from scipy.stats import norm

    assert norm.cdf(delta_a / math.sqrt(2.0)) == pytest.approx(0.78, abs=1e-12)
    assert norm.cdf(delta_b / math.sqrt(2.0)) == pytest.approx(0.72, abs=1e-12)


def test_single_replicate_is_seed_reproducible_and_fields_are_valid():
    p = protocol()
    cell = CellSpec(0, 250, 0.2, 0.2, 0.8, "MCAR")
    gamma_0 = float(logit(0.8))
    first = run_replicate(
        protocol=p, cell=cell, replicate_id=0, gamma_0=gamma_0, base_seed=20260903
    )
    second = run_replicate(
        protocol=p, cell=cell, replicate_id=1, gamma_0=gamma_0, base_seed=20260903
    )
    assert (
        first["total_positives_m"],
        first["auc_a_full"],
        first["auc_b_full"],
    ) != (
        second["total_positives_m"],
        second["auc_a_full"],
        second["auc_b_full"],
    )
    second = run_replicate(
        protocol=p, cell=cell, replicate_id=0, gamma_0=gamma_0, base_seed=20260903
    )
    assert deterministic_projection(first) == deterministic_projection(second)
    assert first["seed_base"] == 20260903
    assert first["delta_population"] == pytest.approx(0.06, abs=1e-12)
    assert 0.0 <= first["achieved_verification_fraction"] <= 1.0
    if not first["target_undefined"]:
        for regime in ("exact", "bounded", "unrestricted"):
            assert first[f"{regime}_count_contains_m"]
            assert first[f"{regime}_finite_truth_contained"]
            assert first[f"{regime}_paired_within_separate"]
            assert first[f"{regime}_lower_le_upper"]
            assert not first[f"{regime}_unexplained_nan_inf"]


def test_changing_replicate_id_changes_generated_result():
    p = protocol()
    cell = CellSpec(0, 250, 0.2, 0.2, 0.8, "MCAR")
    gamma_0 = float(logit(0.8))
    first = run_replicate(
        protocol=p, cell=cell, replicate_id=0, gamma_0=gamma_0, base_seed=20260903
    )
    second = run_replicate(
        protocol=p, cell=cell, replicate_id=1, gamma_0=gamma_0, base_seed=20260903
    )
    assert deterministic_projection(first) != deterministic_projection(second)


def test_target_undefined_branch_retains_replicate_without_intervals():
    p = protocol()
    cell = CellSpec(0, 250, 0.0, 0.2, 0.4, "MCAR")
    row = run_replicate(
        protocol=p,
        cell=cell,
        replicate_id=7,
        gamma_0=float(logit(0.4)),
        base_seed=20260903,
    )
    assert row["total_positives_m"] == 0
    assert row["target_undefined"]
    assert row["replicate_status"] == "target_undefined"
    assert row["warning_code"] == "TARGET_UNDEFINED"
    assert np.isnan(row["delta_full_n"])
    for regime in ("exact", "bounded", "unrestricted"):
        assert row[f"{regime}_count_set_size"] == 0
        assert np.isnan(row[f"{regime}_paired_lower"])


def test_complete_case_undefined_does_not_stop_paired_analysis():
    p = protocol()
    cell = CellSpec(0, 250, 0.2, 0.2, 0.4, "MCAR")
    row = run_replicate(
        protocol=p,
        cell=cell,
        replicate_id=3,
        gamma_0=-100.0,
        base_seed=20260903,
    )
    assert not row["target_undefined"]
    assert row["cc_undefined"]
    assert row["warning_code"] == "CC_UNDEFINED"
    for regime in ("exact", "bounded", "unrestricted"):
        assert row[f"{regime}_finite_truth_contained"]
        assert row[f"{regime}_paired_within_separate"]


def test_simulation_checkpoint_validation_on_small_frame():
    p = protocol()
    cell = CellSpec(0, 250, 0.2, 0.2, 0.8, "MCAR")
    rows = [
        run_replicate(
            protocol=p,
            cell=cell,
            replicate_id=replicate_id,
            gamma_0=float(logit(0.8)),
            base_seed=20260901,
        )
        for replicate_id in range(2)
    ]
    frame = normalize_frame_types(pd.DataFrame(rows))
    counts = validate_cell_frame(
        frame,
        expected_cell_id=0,
        repetitions=2,
        simulation_base_seed=20260901,
    )
    assert all(value == 0 for value in counts.values())
