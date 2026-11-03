"""Checks for the WDBC download boundary, masked API and frozen stream plan."""
import hashlib

import numpy as np
import pytest

from paired_auc.wdbc import identify, unrestricted_counts, verification_masks, verified_download


def test_cached_data_must_match_official_hash(tmp_path):
    path = tmp_path / "wdbc.data"
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256/size mismatch"):
        verified_download(tmp_path, path.name, {
            "url": "https://archive.ics.uci.edu/unneeded",
            "bytes": 8, "sha256": hashlib.sha256(b"expected").hexdigest(),
        })


def test_shared_uniforms_match_frozen_pcg64_stream():
    probabilities = np.broadcast_to(np.array([.4, .6, .8])[:, None], (3, 569))
    probabilities = np.broadcast_to(probabilities, (3, 3, 569))
    masks = verification_masks(probabilities, 0, 0)
    expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [20261102, 34, 1, 0, 0]
    ))).random(569)
    np.testing.assert_array_equal(masks, expected <= probabilities[0])
    assert np.all(masks[0] <= masks[1]) and np.all(masks[1] <= masks[2])


def test_wdbc_identifier_rejects_exposed_hidden_labels():
    a = np.array([1., 2., 3., 4.])
    b = np.array([4., 2., 3., 1.])
    verified = np.array([True, True, False, False])
    exposed = np.array([0., 1., 0., 1.])
    with pytest.raises(ValueError, match="unknown labels must be NaN"):
        unrestricted_counts(exposed, verified)
    with pytest.raises(ValueError, match="unknown labels must be NaN"):
        identify(a, b, verified, exposed, (2,))


def test_wdbc_identifier_uses_observed_count_range():
    obs = np.array([0., 1., np.nan, np.nan])
    verified = np.isfinite(obs)
    assert tuple(unrestricted_counts(obs, verified)) == (1, 2, 3)
    result = identify([1., 2., 3., 4.], [4., 2., 3., 1.], verified, obs, (1, 2, 3))
    assert result["count_set_size"] == 3
    assert result["paired_within_separate"] and result["lower_le_upper"]
