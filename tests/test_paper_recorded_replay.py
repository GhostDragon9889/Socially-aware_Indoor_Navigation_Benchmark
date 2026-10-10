"""CPU admission gates for recorded-pose visualization, without Kit imports."""
from __future__ import annotations

import copy
import subprocess
import sys

import numpy as np
import pytest

from scripts.replay_paper_pedestrians import (bind_source_path, capture_recorded_frame, common_names,
    configure_recorded_capture, parse_args, stamp_recorded_frame, validate_arrays, validate_scene)


def fixture():
    bodies, dofs = common_names()
    arrays = {"root_pos": np.zeros((8, 2, 3), dtype=np.float32),
              "root_rot": np.broadcast_to(np.array([0, 0, 0, 1], dtype=np.float32), (8, 2, 4)).copy(),
              "body_pos": np.zeros((8, 2, 24, 3), dtype=np.float32),
              "dof_pos": np.zeros((8, 2, 69), dtype=np.float32),
              "episode_ids": np.ones((8, 2), dtype=np.int64), "reset_mask": np.zeros((8, 2), dtype=bool)}
    arrays["reset_mask"][0] = True
    arrays["episode_ids"][4:, 1] = 2
    arrays["reset_mask"][4, 1] = True
    result = {"status": "passed", "actual_control_steps": 8, "state_capture": {"frequency_hz": 30, "body_names": list(bodies),
        "coordinates": "world meters, quaternions XYZW, common SMPL names",
        "reset_mask": "first post-step frame of each new episode; physical termination captured separately"}, "physics": {"dof_names": list(dofs)}}
    return arrays, result


def test_source_times_are_poststep_and_cadence_preserves_resets():
    arrays, result = fixture()
    data = validate_arrays(arrays, result, start_step=2, steps=6, capture_every_steps=2)
    assert data["captured_step_indices"] == [2, 4, 6]
    assert data["captured_source_time_s"] == [3/30, 5/30, 7/30]
    assert data["source_interval_s"] == [2/30, 8/30]
    assert data["video_fps"] == 15
    assert arrays["reset_mask"][4, 1]


def test_d6_declared_backend_names_do_not_change_common_recorded_order():
    arrays, result = fixture()
    _, dofs = common_names()
    result["physics"]["dof_names"] = [name.rsplit("_", 1)[0] + "_x:" + str("xyz".index(name[-1])) for name in reversed(dofs)]
    assert validate_arrays(arrays, result, steps=8)["common_dof_names"] == list(dofs)


@pytest.mark.parametrize("field", ["root_rot", "dof_pos", "body_pos"])
def test_wrong_shape_or_nonfinite_pose_is_rejected(field):
    arrays, result = fixture()
    arrays[field].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_arrays(arrays, result, steps=8)
    arrays, result = fixture()
    arrays[field] = arrays[field][:-1]
    with pytest.raises(ValueError, match="shape"):
        validate_arrays(arrays, result, steps=8)


def test_quaternion_order_body_mapping_and_timestamp_contract_fail_closed():
    arrays, result = fixture()
    arrays["root_rot"][0, 0] *= 2
    with pytest.raises(ValueError, match="unit XYZW"):
        validate_arrays(arrays, result, steps=8)
    arrays, result = fixture()
    for field, value in (("frequency_hz", 60), ("coordinates", "world WXYZ"), ("body_names", list(reversed(common_names()[0])))):
        changed = copy.deepcopy(result)
        changed["state_capture"][field] = value
        with pytest.raises(ValueError):
            validate_arrays(arrays, changed, steps=8)
    arrays["body_pos"][0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="Pelvis"):
        validate_arrays(arrays, result, steps=8)


def test_episode_reset_mismatch_and_interval_overrun_are_rejected():
    arrays, result = fixture()
    with pytest.raises(ValueError, match="exceeds"):
        validate_arrays(arrays, result, start_step=4, steps=5)
    arrays["reset_mask"][4, 1] = False
    with pytest.raises(ValueError, match="Episode"):
        validate_arrays(arrays, result, steps=8)


def test_scene_and_fresh_output_gates(tmp_path):
    with pytest.raises(ValueError, match="cuboid"):
        validate_scene({"cuboids": [{"center": [0, 0, 0], "size": [1, -1, 1]}]})
    with pytest.raises(SystemExit):
        parse_args(["--states", "a", "--source-result", "b", "--output", str(tmp_path)])


def test_source_result_must_identify_the_actual_states_file(tmp_path):
    actual = tmp_path / "physical_states.npz"
    bind_source_path(actual, {"state_capture": {"path": str(actual)}})
    with pytest.raises(ValueError, match="different"):
        bind_source_path(actual, {"state_capture": {"path": str(tmp_path / "other.npz")}})


def test_import_is_cpu_preflight_only_in_a_clean_process():
    completed = subprocess.run([sys.executable, "-c",
        "import sys; import scripts.replay_paper_pedestrians; assert 'torch' not in sys.modules; assert 'isaacsim' not in sys.modules; assert 'omni.timeline' not in sys.modules"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


class FakeTimeline:
    def __init__(self):
        self.current_time = 0.
        self.auto_update = True
        self.playing = False

    def set_auto_update(self, enabled):
        self.auto_update = enabled

    def play(self):
        self.playing = True

    def get_current_time(self):
        return self.current_time


class FakeSettings:
    def __init__(self):
        self.values = {}

    def set_bool(self, path, enabled):
        self.values[path] = enabled


class FakeAppAndAnnotator:
    def __init__(self, timeline, settings, *, ready_after=1, advance_time=False):
        self.timeline, self.settings = timeline, settings
        self.ready_after, self.advance_time = ready_after, advance_time
        self.updates = 0

    def update(self):
        assert self.timeline.playing
        assert not self.timeline.auto_update
        assert self.settings.values["/app/player/playSimulations"] is False
        self.updates += 1
        if self.advance_time:
            self.timeline.current_time += 1/30

    def get_data(self):
        if self.updates < self.ready_after:
            return np.zeros((0,), dtype=np.uint8)
        # Model the actual RGB/RGBA API, rather than duplicating capture logic.
        return np.full((80, 128, 4), 80 + self.updates, dtype=np.uint8)


@pytest.mark.parametrize("warmup, expected_updates", [(False, 3), (True, 48)])
def test_pure_render_app_updates_hold_clock_and_use_current_rgb(warmup, expected_updates):
    timeline, settings = FakeTimeline(), FakeSettings()
    configure_recorded_capture(settings, timeline)
    app = FakeAppAndAnnotator(timeline, settings)
    rgb, record = capture_recorded_frame(app, timeline, settings, app, width=128, height=80, initial_time=0., warmup=warmup)
    assert settings.values["/exts/omni.replicator.core/Orchestrator/enabled"] is False
    assert settings.values["/omni/replicator/disableAnnotatorGate"] is True
    assert app.updates == expected_updates == record["app_updates"]
    assert record["timeline_before_s"] == record["timeline_after_s"] == 0.
    np.testing.assert_array_equal(rgb, np.full((80, 128, 3), 80 + expected_updates, dtype=np.uint8))


def test_capture_can_wait_for_rgb_but_empty_annotator_has_hard_update_bound():
    timeline, settings = FakeTimeline(), FakeSettings()
    configure_recorded_capture(settings, timeline)
    app = FakeAppAndAnnotator(timeline, settings, ready_after=7)
    _, record = capture_recorded_frame(app, timeline, settings, app, width=128, height=80, initial_time=0.)
    assert record["app_updates"] == 7
    missing = FakeAppAndAnnotator(timeline, settings, ready_after=1000)
    with pytest.raises(RuntimeError, match="bounded app updates"):
        capture_recorded_frame(missing, timeline, settings, missing, width=128, height=80, initial_time=0.)
    assert missing.updates == 24


def test_capture_refuses_clock_advance_without_resetting_or_hiding_it():
    timeline, settings = FakeTimeline(), FakeSettings()
    configure_recorded_capture(settings, timeline)
    app = FakeAppAndAnnotator(timeline, settings, advance_time=True)
    with pytest.raises(RuntimeError, match="advanced timeline"):
        capture_recorded_frame(app, timeline, settings, app, width=128, height=80, initial_time=0.)
    assert app.updates == 1
    assert timeline.current_time == 1/30
    with pytest.raises(RuntimeError, match="advanced timeline"):
        capture_recorded_frame(app, timeline, settings, app, width=128, height=80, initial_time=0.)
    assert app.updates == 1


def test_replay_banner_is_permanent_changes_with_source_time_and_leaves_pose_pixels_intact():
    rgb = np.full((240, 1280, 3), 100, dtype=np.uint8)
    early, late = stamp_recorded_frame(rgb, 1/30), stamp_recorded_frame(rgb, 200.)
    assert early.shape == rgb.shape and early.dtype == np.uint8
    assert (early[:50] == 255).any()
    assert not np.array_equal(early[:50], late[:50])
    np.testing.assert_array_equal(early[60:], rgb[60:])
    np.testing.assert_array_equal(rgb, np.full_like(rgb, 100))
    with pytest.raises(ValueError, match="timestamp"):
        stamp_recorded_frame(rgb, float("nan"))
