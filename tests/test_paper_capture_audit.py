"""Synthetic payload regressions, never evidence of actual Kit execution."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest


SPEC = importlib.util.spec_from_file_location(
    "paper_capture_audit", Path(__file__).resolve().parents[1] / "scripts/audit_paper_pedestrian_capture.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def payload(tmp_path, monkeypatch):
    args = {"capture": True, "physics_only": False, "deterministic_evaluation": True,
            "evaluate_steps": 1, "capture_every_steps": 2}
    entry = {"schema": "navisaaclab.live_pedestrian_capture_preflight.v1",
             "parsed_driver_arguments": args,
             "capture_request": {"width": 4, "height": 3, "depth": True}}
    row = {"frame": "frame_000000.png", "control_step": 1, "warmup": True,
           "max_absolute_buffer_deltas": dict.fromkeys(
               ["root_pos", "root_rot", "root_vel", "dof_pos", "dof_vel", "rigid_body_pos"], 0.),
           "physics_step_count": {"before": 4, "after": 4, "delta": 0},
           "native_physx_time_s": {"observed": False, "before": None, "after": None, "delta": None},
           "rng": {key: {"observed": True, "equal": True,
                         "before_sha256": "a" * 64, "after_sha256": "a" * 64}
                   for key in ("torch_cpu_rng", "torch_cuda_rng")}}
    result = {"status": "passed", "arguments": args, "capture_state_diagnostics": [row]}
    (tmp_path / "frames").mkdir()
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(tmp_path / "frames/frame_000000.png")
    (tmp_path / "depth").mkdir()
    np.save(tmp_path / "depth/frame_000000.npy", np.full((3, 4), np.inf, dtype=np.float32))
    (tmp_path / "depth/index.jsonl").write_text(json.dumps({
        "control_step": 1, "frame_index": 0, "file": "depth/frame_000000.npy",
        "shape": [3, 4], "dtype": "float32", "units": "metres", "valid_fraction": 0.,
        "valid_min_m": None, "valid_max_m": None, "actual_synchronization_verified": False}) + "\n")
    (tmp_path / "deployment_entry.json").write_text(json.dumps(entry))
    (tmp_path / "result.json").write_text(json.dumps(result))
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps({"streams": [{"codec_name": "h264", "width": 4, "height": 3,
                                         "avg_frame_rate": "15/1", "nb_frames": "1"}]})))
    return tmp_path, result


def test_missing_clock_stays_unknown_and_infinite_background_is_retained(payload):
    directory, _ = payload
    report = MODULE.audit(directory)
    assert report["unavailable_native_clock_frames"] == 1
    assert report["sensor_frame_synchronization_verified"] is False
    assert report["paper_effect_proven"] is False
    assert np.isinf(np.load(directory / "depth/frame_000000.npy")).all()


@pytest.mark.parametrize("channel", ["state", "counter", "rng"])
def test_capture_cannot_pass_if_rendering_changes_state_clock_or_rng(payload, channel):
    directory, result = payload
    row = result["capture_state_diagnostics"][0]
    if channel == "state":
        row["max_absolute_buffer_deltas"]["dof_pos"] = .01
    elif channel == "counter":
        row["physics_step_count"].update(after=5, delta=1)
    else:
        row["rng"]["torch_cpu_rng"]["equal"] = False
    (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        MODULE.audit(directory)


def test_missing_depth_frame_is_a_failure(payload):
    directory, _ = payload
    (directory / "depth/frame_000000.npy").unlink()
    with pytest.raises(ValueError, match="Depth file set"):
        MODULE.audit(directory)


def test_runtime_arguments_cannot_drift_from_recorded_request(payload):
    directory, result = payload
    result["arguments"]["capture_every_steps"] = 3
    (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Runtime arguments differ"):
        MODULE.audit(directory)


@pytest.mark.parametrize("channel", ["all", "cuda"])
def test_missing_rng_observations_cannot_pass_empty_loop(payload, channel):
    directory, result = payload
    rng = result["capture_state_diagnostics"][0]["rng"]
    if channel == "all":
        rng.clear()
    else:
        rng.pop("torch_cuda_rng")
    (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Missing required RNG"):
        MODULE.audit(directory)


@pytest.mark.parametrize("field,value", [("file", "depth/wrong.npy"), ("units", "millimetres"),
                                         ("valid_fraction", 1.), ("actual_synchronization_verified", True)])
def test_depth_index_must_describe_its_actual_file(payload, field, value):
    directory, _ = payload
    path = directory / "depth/index.jsonl"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Depth index metadata"):
        MODULE.audit(directory)


def test_nonfinite_observed_native_clock_is_rejected(payload):
    directory, result = payload
    result["capture_state_diagnostics"][0]["native_physx_time_s"].update(
        observed=True, before=float("inf"), after=float("inf"), delta=0.)
    (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="native physics clock"):
        MODULE.audit(directory)
