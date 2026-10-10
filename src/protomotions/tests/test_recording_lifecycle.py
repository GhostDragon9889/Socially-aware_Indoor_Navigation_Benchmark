"""Recording cancellation, immutable markers and recoverable export failures."""

from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from protomotions.simulator.base_simulator.config import MarkerState
from protomotions.tests.test_recording_mixin import _Recorder, _identity_quat


def test_cancel_before_first_render_is_safe_and_keeps_output_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder = _Recorder()
    recorder._init_recording_state()
    recorder._cancel_video_record()
    recorder.render()
    assert (tmp_path / "output/renderings").is_dir()
    assert recorder._recorded_motion is None
    assert not recorder._delete_user_viewer_recordings


def test_cancel_then_immediate_start_cleans_old_session_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder = _Recorder()
    recorder._init_recording_state()
    recorder._write_viewport_to_file = lambda filename: Path(filename).write_bytes(b"frame")
    recorder._toggle_video_record()
    recorder.render()
    old = Path(recorder._curr_user_recording_name)
    recorder._cancel_video_record()
    recorder._toggle_video_record()
    recorder.render()
    new = Path(recorder._curr_user_recording_name)
    assert new != old and new.is_dir() and not old.exists()
    assert recorder._user_is_recording and len(recorder._recorded_motion["gts"]) == 1
    recorder.render()
    assert len(recorder._recorded_motion["gts"]) == 2


def test_marker_conversion_and_visibility_do_not_mutate_controller_or_recording_state():
    recorder = _Recorder()
    recorder._init_recording_state()
    recorder.config.w_last = False
    recorder._show_markers = False
    position = torch.tensor([[1., 2., 3.]])
    orientation = torch.tensor([[0., 0., .6, .8]])
    markers = {"target": MarkerState(translation=position.clone(), orientation=orientation.clone(), color=(.1, .2, .3))}
    for _ in range(2):
        recorder._update_markers(markers)
        updated = recorder.updated_markers[-1]["target"]
        torch.testing.assert_close(updated.orientation, torch.tensor([[.8, 0., 0., .6]]))
        assert (updated.translation == -1000000).all()
        assert updated.color == (.1, .2, .3)
    torch.testing.assert_close(markers["target"].translation, position)
    torch.testing.assert_close(markers["target"].orientation, orientation)


def test_stop_then_immediate_start_publishes_previous_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder = _Recorder()
    recorder._init_recording_state()
    recorder._write_viewport_to_file = lambda filename: Path(filename).write_bytes(b"frame")
    class Clip:
        def __init__(self, images, **kwargs): pass
        def write_videofile(self, path, **kwargs): Path(path).write_bytes(b"video")
        def close(self): pass
    module = ModuleType("moviepy")
    module.ImageSequenceClip = Clip
    monkeypatch.setitem(sys.modules, "moviepy", module)
    recorder._toggle_video_record()
    recorder.render()
    old = Path(recorder._curr_user_recording_name)
    recorder._toggle_video_record()
    recorder._toggle_video_record()
    recorder.render()
    new = Path(recorder._curr_user_recording_name)
    previous = torch.load(str(old) + ".motion", weights_only=False)
    assert previous["rigid_body_pos"].shape[0] == 1
    assert Path(str(old) + ".mp4").is_file() and not old.exists()
    assert new != old and new.is_dir()
    assert recorder._user_is_recording and len(recorder._recorded_motion["gts"]) == 1


def _ready_recorder(tmp_path):
    recorder = _Recorder()
    recorder._init_recording_state()
    folder = tmp_path / "recording"
    folder.mkdir()
    recorder._curr_user_recording_name = str(folder)
    recorder._recorded_motion = dict(gts=[torch.zeros(2, 3)], grs=[_identity_quat(2)],
                                   gvs=[], gavs=[], dps=[], dvs=[], contacts=[])
    recorder._recorded_markers = {}
    recorder._recorded_objects = []
    recorder._recorded_projectiles = []
    recorder.terrain = None
    return recorder, folder


def test_encoder_failure_closes_clip_and_preserves_motion_and_raw_frames(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder, folder = _ready_recorder(tmp_path)
    (folder / "0000.png").write_bytes(b"frame")
    closed = []
    class Clip:
        def __init__(self, *args, **kwargs): pass
        def write_videofile(self, *args, **kwargs): raise OSError("encoder failed")
        def close(self): closed.append(True)
    module = ModuleType("moviepy")
    module.ImageSequenceClip = Clip
    monkeypatch.setitem(sys.modules, "moviepy", module)
    with pytest.raises(OSError, match="encoder failed"):
        recorder._finalize_user_recording()
    assert closed == [True]
    assert folder.is_dir() and (folder / "0000.png").is_file()
    state = torch.load(str(folder) + ".motion", weights_only=False)
    assert state["rigid_body_pos"].shape[0] == 1
    assert not recorder._delete_user_viewer_recordings


def test_long_recording_frames_sort_by_frame_number(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder, folder = _ready_recorder(tmp_path)
    for number in [10001, 9999, 10000]:
        (folder / f"{number}.png").write_bytes(b"frame")
    ordered = []
    class Clip:
        def __init__(self, images, **kwargs): ordered.extend(Path(image).name for image in images)
        def write_videofile(self, *args, **kwargs): pass
        def close(self): pass
    module = ModuleType("moviepy")
    module.ImageSequenceClip = Clip
    monkeypatch.setitem(sys.modules, "moviepy", module)
    recorder._finalize_user_recording()
    assert ordered == ["9999.png", "10000.png", "10001.png"]


def test_missing_viewport_frames_preserves_physical_motion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder, folder = _ready_recorder(tmp_path)
    with pytest.raises(RuntimeError, match="No viewport PNG"):
        recorder._finalize_user_recording()
    assert Path(str(folder) + ".motion").is_file()
    assert not recorder._delete_user_viewer_recordings
