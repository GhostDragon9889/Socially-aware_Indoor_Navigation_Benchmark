"""Verify recorded media and motion use the same real control clock."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import pytest

from protomotions.simulator.base_simulator.config import MarkerState
from protomotions.tests.test_recording_mixin import _Recorder, _identity_quat


def test_25hz_recording_encodes_one_second_and_matching_motion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder = _Recorder()
    recorder.dt = 1.0 / 25.0
    recorder._init_recording_state()
    recorder._last_markers_state = {
        "target": MarkerState(translation=torch.zeros(2, 3), orientation=_identity_quat(2))
    }
    counter = 0

    def write_frame(filename):
        nonlocal counter
        Image.fromarray(np.full((32, 32, 3), counter * 9, dtype=np.uint8)).save(filename)
        counter += 1

    recorder._write_viewport_to_file = write_frame
    recorder._toggle_video_record()
    for _ in range(25):
        recorder.render()
    prefix = Path(recorder._curr_user_recording_name)
    recorder._toggle_video_record()
    recorder.render()

    from moviepy.video.io.VideoFileClip import VideoFileClip
    with VideoFileClip(str(prefix) + ".mp4") as clip:
        assert clip.fps == pytest.approx(25.0)
        assert clip.duration == pytest.approx(1.0, abs=0.01)
    motion = torch.load(str(prefix) + ".motion", weights_only=False)
    markers = torch.load(str(prefix) + ".markers.pt", weights_only=False)
    assert motion["fps"] == markers["fps"] == 25.0
    assert motion["rigid_body_pos"].shape[0] == 25
    assert markers["markers"]["target"]["translation"].shape[0] == 25


@pytest.mark.parametrize("dt", [0, -0.1, float("inf"), float("nan")])
def test_invalid_recording_clock_is_rejected(dt):
    recorder = _Recorder()
    recorder.dt = dt
    with pytest.raises(ValueError, match="timestep"):
        recorder._recording_fps()
