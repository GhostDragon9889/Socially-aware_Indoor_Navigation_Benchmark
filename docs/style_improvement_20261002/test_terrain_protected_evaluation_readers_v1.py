"""Future reader gates; fixtures are protocol objects, not new physical runs."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


EVIDENCE = module(
    "protected_future_evidence", HERE / "terrain_protected_evaluation_evidence_v1.py"
)
REG = json.loads(
    (
        HERE / "terrain_protected_evaluation_registration_v1/registration.json"
    ).read_text()
)


def fixture():
    # Reuse only observed old22 aggregate/schema to test admission; no fake new NPZ.
    value = json.loads(
        (
            ROOT
            / "output/diffusion_amp/motion_tracking_mixed_curriculum_step2000_full_validation_v1/result.json"
        ).read_text()
    )
    value["arguments"] = deepcopy(REG["tracking"]["parsed"])
    value["arguments"]["checkpoint_sha256"] = "a" * 64
    return value


@pytest.mark.parametrize(
    "defect",
    [
        "test_split",
        "partial",
        "missing_clip",
        "failed_execution",
        "less_frames",
        "float_frames",
    ],
)
def test_incomplete_or_test_protocol_rejected_before_future_artifact_admission(defect):
    result = fixture()
    if defect == "test_split":
        result["arguments"]["split"] = "test"
    elif defect == "partial":
        result["arguments"]["max_steps"] = 32
    elif defect == "missing_clip":
        result["sequences"].pop()
    elif defect == "failed_execution":
        result["status"] = "failed"
    elif defect == "less_frames":
        result["aggregate"]["observed_metric_steps"] = 11981
    else:
        result["aggregate"]["observed_metric_steps"] = 11982.0
    with pytest.raises(ValueError):
        EVIDENCE.validate_protocol(result, "tracking", REG)


def test_correct_full_protocol_shape_does_not_claim_runtime_admitted():
    assert EVIDENCE.validate_protocol(fixture(), "tracking", REG) is None
    assert REG["pending"]["future_full_payload_admitted"] is False
    assert REG["pending"]["physical_evaluation_started"] is False


def test_exact_frozen_45D_basis_and_existing_reference_keys():
    path = HERE / "pose_diversity_candidate_v1/actual_evaluation_v1/reference_basis.npz"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "b25dffa6612d6202b5faf534381c24a4558f347639b366214ec61f5a2478dcdb"
    )
    with np.load(path, allow_pickle=False) as values:
        assert set(values.files) == {"centers_50", "centers_100"}
        assert values["centers_50"].shape == (50, 45) and values[
            "centers_100"
        ].shape == (100, 45)
        assert all(
            v.dtype == np.float64 and np.isfinite(v).all() for v in values.values()
        )
    source = (HERE / "evaluate_terrain_protected_pose45_v1.py").read_text()
    assert "tool.fit_reference = reuse_reference" in source and "tool.main(" in source
