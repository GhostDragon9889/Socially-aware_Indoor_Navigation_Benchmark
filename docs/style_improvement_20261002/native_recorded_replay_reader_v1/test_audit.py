"""CPU admission tests for actual failure records and meaningful banner errors."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("native_recorded_audit_candidate", HERE/"audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
spec = importlib.util.spec_from_file_location("replay_CPU_banner", ROOT/"scripts/replay_paper_pedestrians.py")
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)


@pytest.mark.parametrize("run", ["native_warehouse_recorded_replay_30s_v1", "native_warehouse_recorded_replay_placeholder_diag_v1"])
def test_actual_zero_frame_failures_stay_failed_with_exact_sources(monkeypatch, run):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    report = audit.verify(ROOT/"output/diffusion_amp"/run)
    assert report["status"] == "failed_execution_recorded"
    assert report["renderer_terminal_status"] == "failed"
    assert report["rendered_frames_as_recorded"] == 0
    assert not report["actual_video_success_verified"]
    assert len(report["source_USD_layers_unchanged"]) == 53
    assert report["renderer_error"]


def test_fake_zero_physics_boolean_not_accepted_as_integer(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = json.loads((ROOT/"output/diffusion_amp/native_warehouse_recorded_replay_30s_v1/result.json").read_text())
    result["physics_steps"] = False
    (tmp_path/"result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="execution count"):
        audit.verify(tmp_path)


@pytest.mark.parametrize("corruption", ["missing", "wrong_time", "wrong_dimensions"])
def test_banner_rejects_missing_wrong_time_and_dimensions(tmp_path, corruption):
    rgb = np.random.default_rng(7).integers(20, 200, (96, 1280, 3), dtype=np.uint8)
    expected = producer.stamp_recorded_frame(rgb, .03333333333333333)
    if corruption == "missing":
        expected = rgb
    elif corruption == "wrong_time":
        expected = producer.stamp_recorded_frame(rgb, 9.)
    else:
        expected = expected[:, :-2]
    path = tmp_path/"frame.png"
    Image.fromarray(expected).save(path)
    with pytest.raises(ValueError, match="banner|dimensions"):
        audit.verify_png(path, {"source_time_s": .03333333333333333}, {"width": 1280, "height": 96}, producer)


def test_frame_uses_post_control_time_and_real_source_episode():
    arrays = {"reset_mask": np.array([[True, False]], dtype=bool), "episode_ids": np.array([[3, 2]], dtype=np.int64)}
    frame = {"frame": 0, "source_step_index": 0, "source_time_s": 1/30, "source_reset_mask": [True, False],
             "source_episode_ids": [3, 2], "timeline_before_s": 0., "timeline_after_s": 0., "app_updates": 48}
    audit.check_frame_metadata(frame, 0, 0, arrays)
    frame["source_time_s"] = 0.
    with pytest.raises(ValueError, match="timestamp"):
        audit.check_frame_metadata(frame, 0, 0, arrays)


def test_source_file_changes_rejected(tmp_path):
    p = tmp_path/"resource.usda"
    p.write_text("#usda 1.0\n")
    original = audit.record(p)
    p.write_text("#usda 1.0\n#changed\n")
    with pytest.raises(ValueError, match="Changed declared"):
        audit.checked_file(original)
