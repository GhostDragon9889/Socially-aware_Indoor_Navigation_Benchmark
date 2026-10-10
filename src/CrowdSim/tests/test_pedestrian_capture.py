"""Regression contracts for bounded rendering and Kit fast shutdown."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from CrowdSim.humanoid.simulate import _capture_frame, _write_video


@pytest.mark.parametrize("original", [True, False])
def test_capture_keeps_physics_clock_frozen_and_restores_setting(monkeypatch, original):
    values = {"/app/player/playSimulations": original}
    ticks = []
    physics_time = 0
    settings = SimpleNamespace(get=values.get, set_bool=lambda key, value: values.update({key: value}))
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)))

    def update():
        nonlocal physics_time
        if values["/app/player/playSimulations"]:
            physics_time += 1
        ticks.append(True)

    env = SimpleNamespace(simulator=SimpleNamespace(
        _sim=SimpleNamespace(physics_manager=SimpleNamespace(forward=lambda: None)),
        _simulation_app=SimpleNamespace(update=update),
    ))
    annotator = SimpleNamespace(get_data=lambda: np.full((8, 8, 4), 127, dtype=np.uint8))
    pixels = _capture_frame(env, annotator, warmup=False)
    assert ticks and len(ticks) < 20
    assert physics_time == 0
    assert values["/app/player/playSimulations"] is original
    assert pixels.shape == (8, 8, 3)

    annotator.get_data = lambda: np.empty((0,), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="RGB"):
        _capture_frame(env, annotator, warmup=False)
    assert values["/app/player/playSimulations"] is original


def test_pedestrian_video_exists_and_keeps_capture_cadence(tmp_path):
    frames = [np.full((32, 32, 3), i * 20, dtype=np.uint8) for i in range(10)]
    path = _write_video(tmp_path, frames, fps=10.0)
    from moviepy.video.io.VideoFileClip import VideoFileClip
    with VideoFileClip(path) as clip:
        assert clip.fps == pytest.approx(10.0)
        assert clip.duration == pytest.approx(1.0, abs=0.01)
