"""Tail, analytic-AUC and original-stream Gamma regression checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import betainc, log_ndtr
from scipy.stats import gamma as gamma_distribution

from paired_auc.gamma import NumericalGenerationError, quantiles, run_one

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config/gamma_protocol.json").read_text())
GRID = json.loads((ROOT / "config/gamma_grid.json").read_text())
LOOKUP = {row["cell_id"]: row for row in json.loads((ROOT / "config/gamma_calibration.json").read_text())}
REFERENCE = json.loads((ROOT / "results/summaries/gamma_regression.json").read_text())


def test_gamma_tail_roundtrip_and_underflow_failure():
    z = np.array([-12., -8., -1., 0., 1., 8., 12.])
    for shape in (2, 4):
        q = quantiles(z, shape)
        recovered = np.where(z <= 0, gamma_distribution.logcdf(q, shape), gamma_distribution.logsf(q, shape))
        np.testing.assert_allclose(recovered, log_ndtr(-np.abs(z)), atol=1e-10, rtol=0)
    with pytest.raises(NumericalGenerationError):
        quantiles(np.array([100.]), 2)


def test_gamma_analytic_auc_parameters():
    for model in ("A", "B"):
        parameters = CFG["scale_parameters"][model]
        actual = 1 - betainc(CFG["shape1"], CFG["shape0"], 1 / (1 + parameters["theta_1"]))
        assert abs(actual - parameters["target_auc"]) <= 1e-12


@pytest.mark.parametrize("record", REFERENCE, ids=lambda row: f"cell{row['cell_id']}_rep{row['replicate_id']}")
def test_gamma_original_stream_and_endpoints(record):
    cell = GRID[record["cell_id"]]
    row, arrays = run_one(cell, record["replicate_id"], LOOKUP[cell["cell_id"]]["gamma_0"], stage=4, protocol=CFG)
    digest = hashlib.sha256()
    for key in ("Y", "scores_a", "scores_b", "verified", "observed_labels"):
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    assert digest.hexdigest() == record["arrays_sha256"]
    for key, expected in record["fields"].items():
        assert row[key] == pytest.approx(expected, abs=1e-12, rel=1e-12)
